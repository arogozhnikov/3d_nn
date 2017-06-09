fragment_shader = `
precision highp float;
uniform vec2 resolution;
uniform vec3 cameraPos;
uniform vec3 cameraDir;
uniform float surface_level;
uniform mat4 weights_W[2];
uniform vec4 weights_V[2];

const float EPS = 0.01;
const float OFFSET = EPS * 10.0;
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
	// sign(sin(length(p) * 10.))
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
	
	for ( int i = 0; i < 64; i++ ) {
		dist = evaluate_nn( p ) - surface_level;

		if ( floor(oldDist) != floor(dist) ) {
			factor *= - 0.5;
			min_step = 0.;
		} else {
			factor *= 1.2;
		}

		factor = abs(factor);
		factor = min(factor, 3.);
		if ( original_dist != floor(dist) ) {
			factor = - factor;
		}

		oldDist = dist;
		depth += (abs(dist - floor(dist + 0.5)) + min_step) * factor;
		p = origin + depth * ray;
		if ( abs(dist - floor(dist + 0.5)) < EPS ) break;
	}

	if (length(p) > blind_radius) {
		// killing outside radius
		// return vec3(1, 0, 0);
		discard;
	}

	// hit check and calc color
	vec3 color;
	if ( abs(dist - floor(dist + 0.5)) < EPS ) {
		vec3 normal = getNormal(p);
		float diffuse = clamp( dot( lightDir, normal ), 0.2, 1.0 );
		float specular = pow( clamp( dot( reflect( lightDir, normal ), ray ), 0.0, 1.0 ), 10.0 ) + 0.2;
		color = ( sceneColor( p ).rgb * diffuse + vec3( 0.8 ) * specular ) ;
	} else {
		// return vec3(0, 1, 0);
		discard;
	}
	return color; 
}

void main(void) {
	// fragment position
	vec2 p = ( gl_FragCoord.xy * 2.0 - resolution ) / min( resolution.x, resolution.y );
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
`

var raymarch_vertex_shader = `
attribute vec3 position;
void main(void) {
	gl_Position = vec4(position, 1.0);
}
`

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
	float step = index * 0.02;
	p = vec3(mod(position[0], 100.) - 10., position[1], position[2]) * 1.2 ;
	for (int i=0; i < 20; i++) {
		p += normalize(getGradient(p)) * step;
	}
	// int index_ = int(index);
	// float step = 0.01 / 2.;
	// p = vec3(mod(position[0], 100.) - 10., position[1], position[2]) * 1.2 ;
	// for (int i=0; i < 2 * index_; i++) {
	// 	p += normalize(getGradient(p)) * step;
	// }
	value = evaluate_nn(p);
	gl_Position = projectionMatrix * modelViewMatrix * vec4( p , 1.0 );
	screen_position = gl_Position;
}
`
	
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
	
	// gl_FragColor = vec4(0.5 + value / 5., (0.5 + value / 3.) * 0.3, 0.5 - value / 3., exp(-10. * time_delta) * 0.4 );
	gl_FragColor = vec4(0.5 + value / 10., 0.25, 0.5 - value / 4.,  exp(-7. * time_delta) * 0.4 );
	// gl_FragColor = vec4(1., 1., 1., exp(-5. * time_delta) );
	return;
}
`

var position_camera, dummy_camera, dummy_scene, controls, renderer;
var dummy_geometry, dummy_material, dummy_mesh, lines_geometry, lines_material;

var mouse = new THREE.Vector2( 0.5, 0.5 );
var canvas_size = 512;
var canvas;
var stats;
var clock = new THREE.Clock();
var config = {
	// saveImage: function() {
	// 	renderer.render( scene, dummy_camera );
	// 	window.open( canvas.toDataURL() );
	// },
	// freeCamera: false,
	resolution: '512'
};


function init() {
	dummy_scene = new THREE.Scene();
	lines_scene = new THREE.Scene();
	dummy_camera = new THREE.Camera();
	position_camera = new THREE.PerspectiveCamera( 45 * 2 /* degrees */, 1. /* aspect */, 1 /*near plane*/, 1000 /* far plane */ );
	position_camera.lookAt( new THREE.Vector3( 0.0, 0.0, 0.0 ) );
	dummy_geometry = new THREE.PlaneBufferGeometry( 2.0, 2.0 );

	// var weights_W = [new THREE.Matrix4(), new THREE.Matrix4()];
	// var weights_V = [new THREE.Vector4(), THREE.Vector4()];
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
	canvas.addEventListener( 'mousemove', onMouseMove );
	document.getElementById('canvas_container').appendChild( canvas ); 

	// TODO convert to a single line
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


	var gui = new dat.GUI();
	gui.add( config, 'resolution', [ '256', '512', '800' ] ).name( 'Resolution' ).onChange( function( value ) {
		canvas.width = value;
		canvas.height = value;
		renderer.setSize( canvas.width, canvas.height );
	} );
	stats = new Stats();
	document.body.appendChild( stats.dom );
}


function render( timestamp ) {
	stats.begin();

	// update camera
	var _y = 3 * (mouse.y - 0.5);
	var _x = 6 * mouse.x;
	position_camera.position.set( 10. * Math.cos(_x) * Math.cos(_y), 10. * Math.sin(_y), 10. * Math.sin(_x) * Math.cos(_y));
	position_camera.lookAt( new THREE.Vector3( 0.0, 0.0, 0.0 ) );


	var weights_W = [new THREE.Matrix4(), new THREE.Matrix4()];
	weights_W[0].elements = Array.prototype.concat.apply([], [W[0], W[1], W[2], W[3]]);
	weights_W[1].elements = Array.prototype.concat.apply([], [W[4], W[5], W[6], W[7]]);
	var weights_V = [new THREE.Vector4(V[0][0], V[0][1], V[0][2], V[0][3]), new THREE.Vector4(V[0][4], V[0][5], V[0][6], V[0][7])];	
	dummy_material.uniforms.weights_W.value = weights_W;
	dummy_material.uniforms.weights_V.value = weights_V;

	dummy_material.uniforms.resolution.value = new THREE.Vector2( canvas.width, canvas.height );
	dummy_material.uniforms.cameraPos.value = position_camera.getWorldPosition();
	dummy_material.uniforms.cameraDir.value = position_camera.getWorldDirection();
	dummy_material.uniforms.surface_level.value = timestamp / 5000.;
	renderer.autoClear = false;
	renderer.render( lines_scene, position_camera );
	renderer.render( dummy_scene, dummy_camera );
	// lines_renderer.render( lines_scene, camera );
	stats.end();
	requestAnimationFrame( render );
}


function onMouseMove( e ) {
	mouse.x = e.offsetX / canvas.width;
	mouse.y = e.offsetY / canvas.height;
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
	if (elem.addEventListener) {
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
	} else { // IE8-
		text.attachEvent("onmousewheel", handler);
	}
}


var control_cells = [];

function updateWeight(cell, delta){
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
	for(var i = 0; i < n_hidden; i++) {
		control_cells[i + 1][n_input].innerHTML = ' &rarr; h<sub>'+ (i + 1) +'</sub> &rarr;';
	}
	control_cells[0][n_input + 1].innerHTML = 'output <br /> &uarr; '
	control_cells[0][n_input + 1].colSpan = "2";


	// first connection
	for(var i=0; i < n_input; i++) {
		for(var j=0; j < n_hidden; j++) {
			var cell = control_cells[j+1][i];
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

