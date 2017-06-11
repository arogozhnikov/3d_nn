var fragment_shader = `
precision highp float;
uniform vec2 resolution;
uniform vec3 cameraPos;
uniform vec3 cameraDir;
uniform float surface_level;
uniform mat4 weights_W[2];
uniform vec4 weights_V[2];

const float EPS = 0.01;
const vec3 lightDir = vec3(0, 1, 0);
const float blind_radius = 5.;

vec4 tanh(vec4 x){
	vec4 exp_2 = exp(2.0 * x);
	return (exp_2 - 1.0) / (exp_2 + 1.0);
}

float evaluate_nn( vec3 p ) {
	float result = 0.;
	vec4 ext_position = vec4(p, 1);
	for (int i = 0; i < 2; i++ ) {
		vec4 res1 = ext_position * weights_W[i];
		result += dot(tanh(res1), weights_V[i]);
	}
	return result;
}

vec4 sceneColor( vec3 p ) {
	return vec4((p + sign(sin(length(p) * 10.))) * 0.1 + 0.5, 1.);
}

vec3 getNormal( vec3 p ) {
	return normalize(vec3(
		evaluate_nn(p + vec3( EPS, 0.0, 0.0 ) ) - evaluate_nn(p + vec3( -EPS, 0.0, 0.0 ) ),
		evaluate_nn(p + vec3( 0.0, EPS, 0.0 ) ) - evaluate_nn(p + vec3( 0.0, -EPS, 0.0 ) ),
		evaluate_nn(p + vec3( 0.0, 0.0, EPS ) ) - evaluate_nn(p + vec3( 0.0, 0.0, -EPS ) )
	));
}

vec3 getRayColor( vec3 origin, vec3 ray) {
	// moving right to the target sphere
	// origin + alpha * ray has norm of blind_radius
	// length(origin)^2 + 2 * alpha dot(origin, ray) + alpha^2 * 1 = blind_radius^2
	float b = dot(origin, ray);
	float c = pow(length(origin), 2.) - blind_radius * blind_radius; 
	float alpha = - b + sign(b) * pow(b * b - c, 0.5);
	origin = origin + alpha * ray;
	vec3 p = origin;

	float dist;
	float depth = 0.0;
	float oldDepth = 0.0;
	
	float oldDist = evaluate_nn( p ) - surface_level;
	float original_sign = sign(oldDist);
	float original_dist = floor(oldDist);

	float factor = 0.5;
	float min_step = 0.02;
	bool intersected = false;
	
	// marching-like loop to find intersection
	for ( int i = 0; i < 64; i++ ) {
		dist = evaluate_nn( p ) - surface_level;

		if ( floor(oldDist) != floor(dist) ) {
			intersected = true;
			break;
		} 
		
		factor *= 1.2;
		factor = min(factor, 3.);
		
		oldDist = dist;
		oldDepth = depth;
		depth += (abs(dist - floor(dist + 0.5)) + min_step) * factor;
		p = origin + depth * ray;
	}

	float newDepth;
	float newDist;

	// making it precise with hord-like method
	if(intersected) {
		float target_value = floor(max(oldDist, dist));
		for( int i=0; i < 20; i++) {
			newDepth = depth - (depth - oldDepth) / (dist - oldDist) * (dist - target_value);
			newDist = evaluate_nn( origin + newDepth * ray ) - surface_level;
			if((newDist - target_value) * (oldDist - target_value) < 0.){
				dist = newDist;
				depth = newDepth;
			} else {
				oldDist = newDist;
				oldDepth = newDepth;
			}
		}
		dist = newDist;
		p = origin + newDepth * ray;
	}

	// left sphere
	if (length(p) > blind_radius) {
		discard;
	}

	// hit check and calc color
	vec3 color;
	if ( abs(dist - floor(dist + 0.5)) < EPS ) {
		vec3 normal = getNormal(p);
		if(dot(normal, ray) > 0.){
			normal = - normal;
		}
		float diffuse = clamp( dot( lightDir, normal ), 0.3, 1.0 );
		float specular = pow( clamp( dot( reflect( lightDir, normal ), ray ), 0.0, 1.0 ), 10.0 ) + 0.2;
		color = ( sceneColor( p ).rgb * diffuse + vec3( 0.7 ) * specular ) ;
	} else {
		discard;
	}
	return color; 
}

void main(void) {
	// fragment position
	vec2 p = ( gl_FragCoord.xy * 2.0 - resolution ) / min( resolution.x, resolution.y ) / 2.;
	// camera and ray
	vec3 cPos  = cameraPos;
	vec3 cDir  = cameraDir;
	vec3 cSide = normalize( cross( cDir, vec3( 0.0, 1.0 ,0.0 ) ) );
	vec3 cUp   = normalize( cross( cSide, cDir ) );

	float targetDepth = 1.0;
	vec3 ray = normalize( cSide * p.x + cUp * p.y + normalize(cDir) * targetDepth );

	// dummy check for out of region. Ray is normalized
	float shortest_distance = length(cPos - ray * dot(ray, cPos));
	if (shortest_distance > blind_radius) {
		discard;
	}
	
	vec3 color = getRayColor( cPos, ray);

	gl_FragColor = vec4(color, 0.6);
	// webGL doesn't fully support gl_FragDepth. ARGH!
}
`;

var raymarch_vertex_shader = `
attribute vec3 position;
void main(void) {
	gl_Position = vec4(position, 1.0);
}
`;

var lines_vertex_shader = `
// attribute vec3 position;
varying float value;
varying vec3 p;
varying vec4 screen_position;

uniform mat4 weights_W[2];
uniform vec4 weights_V[2];

const float EPS = 0.01;
const float OFFSET = EPS * 10.0;
const vec3 lightDir = vec3(0, 1, 0);

vec4 tanh(vec4 x){
	vec4 exp_2 = exp(2.0 * x);
	return ((exp_2 - 1.0)/(exp_2 + 1.0));
}

float evaluate_nn( vec3 p ) {
	float result = 0.;
	vec4 ext_position = vec4(p, 1);
	for (int i = 0; i < 2; i++ ) {
		vec4 res1 = ext_position * weights_W[i];
		result += dot(tanh(res1), weights_V[i]);
	}
	return result;
}

vec3 getGradient( vec3 p ) {
	return vec3(
		evaluate_nn(p + vec3( EPS, 0.0, 0.0 ) ) - evaluate_nn(p + vec3( -EPS, 0.0, 0.0 ) ),
		evaluate_nn(p + vec3( 0.0, EPS, 0.0 ) ) - evaluate_nn(p + vec3( 0.0, -EPS, 0.0 ) ),
		evaluate_nn(p + vec3( 0.0, 0.0, EPS ) ) - evaluate_nn(p + vec3( 0.0, 0.0, -EPS ) )
	);
}

void main(void) {
	float index = floor(position[0] / 100.);
	float step = (index - 4.) * 0.02;
	p = vec3(mod(position[0], 100.) - 10., position[1], position[2]);
	for (int i=0; i < 20; i++) {
		p += normalize(getGradient(p)) * step;
	}
	value = evaluate_nn(p);
	gl_Position = projectionMatrix * modelViewMatrix * vec4( p , 1.0 );
	screen_position = gl_Position;
} 
`;
	
var lines_fragment_shader = `
uniform vec3 color;
uniform float surface_level;
varying float value;
varying vec3 p;
varying vec4 screen_position;

void main() {
	if (length(p) > 5.) {
		discard;
	}
	float time_delta = fract(surface_level - value); 
	
	gl_FragColor = vec4(0.5 + value / 10., 0.25 - value / 10., 0.5 - value / 4.,  exp(-7. * time_delta) * 0.4 );
	return;
}
`;

var position_camera, dummy_camera, dummy_scene, controls, renderer;
var dummy_geometry, dummy_material, dummy_mesh, lines_geometry, lines_material;
var animation_loop = 5; // seconds

var capturer = null;

var mouse = new THREE.Vector2( 0.5, 0.5 );
var canvas_size = 400;
var canvas;
// var stats;
// var clock = new THREE.Clock();
// var config = {
	// saveImage: function() {
	// 	renderer.render( scene, dummy_camera );
	// 	window.open( canvas.toDataURL() );
	// },
	// freeCamera: false,
	// resolution: canvas_size
// };

var animate_control = document.getElementById('animate_checkbox');
var level_control = document.getElementById('animate_level');
var azimuth_control = document.getElementById('camera_azimuth_control');
var altitude_control = document.getElementById('camera_altitide_control');


animate_control.onchange = animate_control.oninput = function(){
	var group = document.getElementById('surface-control-group');
	if(animate_control.checked){
		group.classList.add('invisible');
	} else {
		group.classList.remove('invisible');
	}
}

function init() {
	dummy_scene = new THREE.Scene();
	lines_scene = new THREE.Scene();
	dummy_camera = new THREE.Camera();
	position_camera = new THREE.PerspectiveCamera( Math.atan(1 / 2.) * (180. / Math.PI) * 2 /* degrees */, 1. /* aspect */, 1 /*near plane*/, 1000 /* far plane */ );
	dummy_geometry = new THREE.PlaneBufferGeometry( 2.0, 2.0 );

	var uniforms = {
			resolution: { value: new THREE.Vector2( canvas_size, canvas_size ) },
			cameraPos:  { value: position_camera.getWorldPosition() },
			cameraDir:  { value: position_camera.getWorldDirection() },
			weights_W : { type: "m4v", value: false }, // Matrix4 array, will be passed later
			weights_V : { type: "v4v", value: false }, // Vector4 array, will be passed later
			surface_level: {value: 0.}
	};

	dummy_material = new THREE.RawShaderMaterial( {
		uniforms: uniforms,
		vertexShader: raymarch_vertex_shader,
		fragmentShader: fragment_shader,
		transparent: true,
	} );
	dummy_mesh = new THREE.Mesh( dummy_geometry, dummy_material );
	dummy_scene.add( dummy_mesh );

	renderer = new THREE.WebGLRenderer();
	renderer.setPixelRatio( window.devicePixelRatio );
	renderer.setSize( canvas_size, canvas_size );

	canvas = renderer.domElement;
	canvas.addEventListener( 'mousemove', function( e ) {
		mouse.x = e.offsetX / canvas.width;
		mouse.y = e.offsetY / canvas.height;
	});
	canvas.addEventListener( 'mouseleave', function(){
		mouse.x = 0.5;
		mouse.y = 0.5;
	});
	canvas.id = 'main_canvas';
	document.getElementById('canvas_container').appendChild( canvas ); 

	// TODO convert set of lines to a single line
	lines_material = new THREE.ShaderMaterial( {
		uniforms:       uniforms,
		vertexShader:   lines_vertex_shader,
		fragmentShader: lines_fragment_shader,
		blending:       THREE.AdditiveBlending,
		depthTest:      false,
		depthWrite:		false,
		transparent:    true, 
		wireframe: 		true,
	});

	var size = 4;
	for(var x_i=-size; x_i < size; x_i++) {
		for (var y_i=-size; y_i < size; y_i++) {
			for (var z_i=-size; z_i < size; z_i++) {
				var lines_geometry = new THREE.Geometry();
				for (var i = 0; i < 10; i++) {
					lines_geometry.vertices.push( new THREE.Vector3( x_i + 10 + i * 100, y_i, z_i ) );	
				}
				lines_scene.add( new THREE.Line( lines_geometry, lines_material ) );
			}
		}
	}	

	helper = new THREE.AxisHelper(5.);
	helper.material.transparent = true;
	helper.material.opacity = 0.6;
	lines_scene.add(helper);

	var show_surface_control = document.getElementById('show_surface_control')
	show_surface_control.onchange = function(){ dummy_mesh.material.visible = show_surface_control.checked; }
	var show_sparks_control = document.getElementById('show_sparks_control')
	show_sparks_control.onchange = function(){ lines_material.visible = show_sparks_control.checked; }
	var show_axes_control = document.getElementById('show_axes_control')
	show_axes_control.onchange = function(){ helper.material.visible = show_axes_control.checked; }


	// var gui = new dat.GUI();
	// gui.add( config, 'resolution', [ '256', '512', '800' ] ).name( 'Resolution' ).onChange( function( value ) {
	// 	canvas.width = value;
	// 	canvas.height = value;
	// 	renderer.setSize( canvas.width, canvas.height );
	// } );
	// stats = new Stats();
	// document.body.appendChild( stats.dom );
}


function render( timestamp, skip_request ) {
	// stats.begin();

	// update camera, always look at center
	var _y = 3.14 * (mouse.y - 0.5) / 2. + parseFloat(altitude_control.value);
	var _x = 3 * mouse.x + parseFloat(azimuth_control.value);
	var r = 12.;
	position_camera.position.set( r * Math.cos(_x) * Math.cos(_y), r * Math.sin(_y), r * Math.sin(_x) * Math.cos(_y));
	position_camera.lookAt( new THREE.Vector3( 0.0, 0.0, 0.0 ) );


	var weights_W = [new THREE.Matrix4(), new THREE.Matrix4()];
	weights_W[0].elements = Array.prototype.concat.apply([], [W[0], W[1], W[2], W[3]]);
	weights_W[1].elements = Array.prototype.concat.apply([], [W[4], W[5], W[6], W[7]]);
	var weights_V = [new THREE.Vector4(V[0][0], V[0][1], V[0][2], V[0][3]), new THREE.Vector4(V[0][4], V[0][5], V[0][6], V[0][7])];	
	dummy_material.uniforms.weights_W.value = weights_W;
	dummy_material.uniforms.weights_V.value = weights_V;

	dummy_material.uniforms.resolution.value = new THREE.Vector2( renderer.getSize().width, renderer.getSize().width );
	dummy_material.uniforms.cameraPos.value = position_camera.getWorldPosition();
	dummy_material.uniforms.cameraDir.value = position_camera.getWorldDirection();
	var level_value = level_control.value;
	if ( animate_control.checked ) {
		level_value = (timestamp / (1000. * animation_loop) ) % 1;
	}

	dummy_material.uniforms.surface_level.value = level_value;
	renderer.autoClear = false;
	renderer.clear();
	renderer.render( lines_scene, position_camera );
	renderer.render( dummy_scene, dummy_camera );

	if( capturer ) capturer.capture( renderer.domElement );
	// stats.end();
	if( !skip_request) requestAnimationFrame( render );
}



var n_input = 3 + 1
var n_hidden = 8

function makeMatrix(m, n){
	var result = new Array(m);
	for (var i = 0; i < m; i++) {
		result[i] = new Array(n);
	}
	return result
}

function initWithRandomUniform(matrix, std){
	for(var i=0; i < matrix.length; i++){
		for(var j=0; j < matrix[i].length; j++){
			matrix[i][j] = (Math.random() * 2 - 1) * std;
		}
	}
}

var W = makeMatrix(n_hidden, n_input);
var V = makeMatrix(1, n_hidden);
var weights = [W, V];

function initWeightsRandom(){
	for(var i=0; i < weights.length; i++){
		initWithRandomUniform(weights[i], 1.);
	}
	var weight_cells = document.getElementsByClassName("weight-control");
	for (var i = 0; i < weight_cells.length; i++) {
		var cell = weight_cells[i];
		updateWeight(cell, 0.);
	}
}

initWeightsRandom();
document.getElementById('randomize_button').onclick = initWeightsRandom;

function addOnWheel(elem, handler) {
	if ('onwheel' in document) {
		// IE9+, FF17+
		elem.addEventListener("wheel", handler);
	} else if ('onmousewheel' in document) {
		// a bit deprecated
		elem.addEventListener("mousewheel", handler);
	} else {
		// 3.5 <= Firefox < 17
		elem.addEventListener("MozMousePixelScroll", handler);
	}
}


var control_cells = [];

var lastedit_timestamp = +new Date();

function updateWeight(cell, delta){
	var timestamp = + new Date();
	if ((delta != 0) && (timestamp - lastedit_timestamp < 300)) return;
	// it is hard to work well on different devices
	delta = - Math.sign(delta) * 0.15;
	var layer = cell.position_layerij[0];
	var i = cell.position_layerij[1];
	var j = cell.position_layerij[2];
	weights[layer][j][i] = Math.max(-1, Math.min(1, weights[layer][j][i] + delta));
	var value = (weights[layer][j][i] + 1) / 2.;
	value = Math.round(100 * value);
	cell.style.backgroundColor = "rgb(" + (50 + value) + ", 50," + (150 - value) + ")";
}

function createControlsTable() {
	var n_columns = n_input + 1 + 1 + 1;
	var n_rows = n_hidden + 1;
    var table = document.createElement('table');
	table.classList.add('control-table');
	for(var row = 0; row < n_rows; row++) {
		control_cells[row] = [];
	    var tablerow = table.insertRow(row);
		for(var col=0; col < n_columns; col++) {
			control_cells[row][col] = tablerow.insertCell();
		}
	}
	control_cells[0][0].innerHTML = 'x <br /> &darr;';
	control_cells[0][1].innerHTML = 'y <br /> &darr;';
	control_cells[0][2].innerHTML = 'z <br /> &darr;';
	control_cells[0][3].innerHTML = '1 <br /> &darr;';
	control_cells[0][0].style.color = '#fbb';
	control_cells[0][1].style.color = '#bbf';
	control_cells[0][2].style.color = '#bfb';
	for(var i = 0; i < n_hidden; i++) {
		control_cells[i + 1][n_input].innerHTML = ' &rarr; h<sub>'+ (i + 1) +'</sub> &rarr;';
	}
	control_cells[0][n_input + 1].innerHTML = 'output <br /> &uarr; '
	control_cells[0][n_input + 1].colSpan = "2";


	// first connection
	for(var i=0; i < n_input; i++) {
		for(var j=0; j < n_hidden; j++) {
			// swapping y and z for users pleasure
			var cell_position = [0, 2, 1, 3][i];
			var cell = control_cells[j+1][cell_position];
			cell.position_layerij = [0, i, j]; 
			cell.classList.add('weight-control');
		}
	}
	// second connection
	for(var i = 0; i < n_hidden; i++) {
		var cell = control_cells[i+1][n_input + 1];
		cell.position_layerij = [1, i, 0]; 
		cell.classList.add('weight-control');
	}

	document.getElementById('control_container').appendChild(table);

	// handlers for mousewheel
	var weight_cells = document.getElementsByClassName("weight-control");
	for (var i = 0; i < weight_cells.length; i++) {
		var cell = weight_cells[i];
		updateWeight(cell, 0.);
		addOnWheel(cell, function(e) {
			e.stopPropagation();
			e.preventDefault();
			var delta = e.deltaY || e.detail || e.wheelDelta;
			updateWeight(e.target, delta * 0.05);
		});
	} 
}

createControlsTable();
init();
render();

function saveAndDownloadVideo(format){
	var canvas = document.getElementById('main_canvas');
	var context = canvas.getContext('webgl');

	capturer = new CCapture( { 
		format: format, 
		framerate: 10,
		verbose: true,
		workersPath: 'utils/',
		name: 'neural_network_3d',
		quality: 90
	} );
	capturer.start();
	for(var timestamp = 0; timestamp < animation_loop * 1000; timestamp += 100){
		render(timestamp, true);
	}
	capturer.stop();
	capturer.save();
	capturer = null;
}

document.getElementById('makegif_button').onclick = function(){saveAndDownloadVideo('gif')};
document.getElementById('makemov_button').onclick = function(){saveAndDownloadVideo('webm')};