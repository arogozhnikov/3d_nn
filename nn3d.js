fragment_shader = `
precision highp float;
uniform vec2 resolution;
uniform vec3 cameraPos;
uniform vec3 cameraDir;
uniform float time;
uniform mat4 nnWeights1[2];
uniform vec4 nnWeights2[2];

const float EPS = 0.01;
const float OFFSET = EPS * 10.0;
const vec3 lightDir = vec3(0, 1, 0);
const float blind_radius = 5.;

vec4 tanh(vec4 x){
	vec4 exp_2 = exp(2.0 * x);
	return (exp_2 - 1.0) / (exp_2 + 1.0);
}

float getValue(float time) {
	return time;
}

float estimate_nn( vec3 p ) {
	float result = 0.;
	vec4 ext_position = vec4(p, 1);
	for (int i = 0; i < 2; i++ ) {
		vec4 res1 = ext_position * nnWeights1[i];
		result += dot(tanh(res1), nnWeights2[i]);
	}
	return result;
}

vec4 sceneColor( vec3 p ) {
	return vec4(p * 0.1 + 0.5, 1.);
}

vec3 getNormal( vec3 p ) {
	return normalize(vec3(
		estimate_nn(p + vec3( EPS, 0.0, 0.0 ) ) - estimate_nn(p + vec3( -EPS, 0.0, 0.0 ) ),
		estimate_nn(p + vec3( 0.0, EPS, 0.0 ) ) - estimate_nn(p + vec3( 0.0, -EPS, 0.0 ) ),
		estimate_nn(p + vec3( 0.0, 0.0, EPS ) ) - estimate_nn(p + vec3( 0.0, 0.0, -EPS ) )
	));
}

vec3 getRayColor( vec3 origin, vec3 ray, out vec3 p, out vec3 normal, out bool hit ) {
	// marching loop
	vec4 nn_result;

	// moving right to the target
	// origin + alpha * ray has norm of blind_radius
	// length(origin)^2 + 2 * alpha dot(origin, ray) + alpha^2 * 1 = blind_radius^2
	float b = dot(origin, ray);
	float c = pow(length(origin), 2.) - blind_radius * blind_radius; 
	float alpha = - b + sign(b) * pow(b * b - c, 0.5);
	origin = origin + alpha * ray;
	p = origin;

	float dist;
	float depth = 0.0;
	float oldDepth = 0.0;
	
	float oldDist = estimate_nn( p ) + getValue(time);
	float original_sign = sign(oldDist);
	float original_dist = floor(oldDist);

	float factor = 0.5;
	float min_step = 0.02;
	
	for ( int i = 0; i < 64; i++ ) {
		dist = estimate_nn( p ) + getValue(time);

		if ( floor(oldDist) != floor(dist) ) {
			factor *= - 0.5;
			min_step = 0.;
		} else {
			factor *= 1.2;
		}

		factor = abs(factor);
		if ( original_dist != floor(dist) ) {
			factor = - factor;
		}
		factor = min(factor, 3.);

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

	// if (sin(length(p) * 10.) > 0.) {
	// 	discard;
	//}

	// hit check and calc color
	vec3 color;
	if ( abs(dist - floor(dist + 0.5)) < EPS ) {
		normal = getNormal(p);
		float diffuse = clamp( dot( lightDir, normal ), 0.2, 1.0 );
		float specular = pow( clamp( dot( reflect( lightDir, normal ), ray ), 0.0, 1.0 ), 10.0 ) + 0.2;
		// float shadow = 0.; 
		// getShadow( p + normal * OFFSET, lightDir );
		color = ( sceneColor( p ).rgb * diffuse + vec3( 0.8 ) * specular ) ; //* max( 0.5, shadow );
		hit = true;
	} else {
		// return vec3(0, 1, 0);
		discard;
	}
	return color; // - pow( clamp( 0.05 * depth, 0.0, 0.6 ), 2.0 );
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

	// dummy check for out of region. ray is normalized
	float shortest_distance = length(cPos - ray * dot(ray, cPos));
	if (shortest_distance > blind_radius) {
		discard;
	}
	
	vec3 q, normal;
	bool hit;

	vec3 color = getRayColor( cPos, ray, q, normal, hit );

	gl_FragColor = vec4(color, 0.6);
}
`

vertex_shader = `
attribute vec3 position;
void main(void) {
	gl_Position = vec4(position, 1.0);
}
`

lines_vertex_shader = `
// attribute vec3 position;
varying float value;
varying vec3 p;
varying vec4 screen_position;

uniform mat4 nnWeights1[2];
uniform vec4 nnWeights2[2];

const float EPS = 0.01;
const float OFFSET = EPS * 10.0;
const vec3 lightDir = vec3(0, 1, 0);

vec4 tanh(vec4 x){
	vec4 exp_2 = exp(2.0 * x);
	return ((exp_2 - 1.0)/(exp_2 + 1.0));
}

float estimate_nn( vec3 p ) {
	float result = 0.;
	vec4 ext_position = vec4(p, 1);
	for (int i = 0; i < 2; i++ ) {
		vec4 res1 = ext_position * nnWeights1[i];
		result += dot(tanh(res1), nnWeights2[i]);
	}
	return result;
}

vec3 getGradient( vec3 p ) {
	return vec3(
		estimate_nn(p + vec3( EPS, 0.0, 0.0 ) ) - estimate_nn(p + vec3( -EPS, 0.0, 0.0 ) ),
		estimate_nn(p + vec3( 0.0, EPS, 0.0 ) ) - estimate_nn(p + vec3( 0.0, -EPS, 0.0 ) ),
		estimate_nn(p + vec3( 0.0, 0.0, EPS ) ) - estimate_nn(p + vec3( 0.0, 0.0, -EPS ) )
	);
}

void main(void) {
	float index = floor(position[0] / 100.);
	float step = index * 0.01;
	p = vec3(mod(position[0], 100.) - 10., position[1], position[2]) * 1.2 ;
	for (int i=0; i < 20; i++) {
		p += normalize(getGradient(p)) * step;
	}
	value = estimate_nn(p);
	gl_Position = projectionMatrix * modelViewMatrix * vec4( p , 1.0 );
	screen_position = gl_Position;
}
`
	
lines_fragment_shader = `
uniform vec3 color;
uniform float time;
varying float value;
varying vec3 p;
varying vec4 screen_position;

void main() {
	if (length(p) > 5.) {
		discard;
	}
	float time_delta = fract(value + time); 
	
	gl_FragColor = vec4(0.5 + value / 5., (0.5 + value / 3.) * 0.3, 0.5 - value / 3., exp(-10. * time_delta) * 0.4 );
	return;
}
`

var position_camera, dummy_camera, scene, controls, renderer;
// var lines_scene;
var geometry, material, mesh;
var lines_canvas, lines_geometry, lines_material, lines_renderer, lines_scene;
var mouse = new THREE.Vector2( 0.5, 0.5 );
var canvas;
var stats;
var clock = new THREE.Clock();
var config = {
	saveImage: function() {
		renderer.render( scene, dummy_camera );
		window.open( canvas.toDataURL() );
	},
	freeCamera: false,
	resolution: '512'
};
init();
render();
function init() {
	scene = new THREE.Scene();
	dummy_camera = new THREE.Camera();
	position_camera = new THREE.PerspectiveCamera( 45 * 2, 500. / 500., 1, 1000 );
	position_camera.lookAt( new THREE.Vector3( 0.0, 0.0, 0.0 ) );
	geometry = new THREE.PlaneBufferGeometry( 2.0, 2.0 );

	lines_scene = new THREE.Scene();

	var nnWeights1 = [new THREE.Matrix4(), new THREE.Matrix4()];
	nnWeights1[0].set(1, 0, -1, 0, 1, 0, 1, -0.1, -1, -1, 0, 1, -0.7, -1, -1, -0.2);
	nnWeights1[1].set(0, -1, 0.3, 1, 0, 1, 0, -1, -1, 0, 1, -1.4, -1, -1, 0, 2);

	var nnWeights2 = [(new THREE.Vector4(2, -1, 2, -1)).multiplyScalar(0.3), (new THREE.Vector4(-0.5, -2 * .7, -1., 0.7)).multiplyScalar(0.3)];
	// console.log(nnWeights2);
	var uniforms = {
			resolution: { value: new THREE.Vector2( 512, 512 ) },
			cameraPos:  { value: position_camera.getWorldPosition() },
			cameraDir:  { value: position_camera.getWorldDirection() },
			nnWeights1 : { type: "m4v", value: nnWeights1 }, // Matrix4 array
			nnWeights2 : { type: "v4v", value: nnWeights2 }, // Vector4 array
			time: {value: 0.}
	};

	material = new THREE.RawShaderMaterial( {
		uniforms: uniforms,
		vertexShader: vertex_shader,
		fragmentShader: fragment_shader,
		transparent: true,
	} );
	mesh = new THREE.Mesh( geometry, material );
	scene.add( mesh );

	renderer = new THREE.WebGLRenderer();
	renderer.setPixelRatio( window.devicePixelRatio );
	renderer.setSize( 512, 512 );
	canvas = renderer.domElement;
	canvas.addEventListener( 'mousemove', onMouseMove );
	// window.addEventListener( 'resize', onWindowResize );
	document.body.appendChild( canvas );


	// single line
	// lines_material = new THREE.LineBasicMaterial({color: 0xffffff});
	lines_material = new THREE.ShaderMaterial( {
		uniforms:       uniforms,
		vertexShader:   lines_vertex_shader,
		fragmentShader: lines_fragment_shader,
		blending:       THREE.AdditiveBlending,
		depthTest:      false,
		depthWrite:		false,
		transparent:    true, 
		wireframe: true,
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
	// renderer
	lines_renderer = new THREE.WebGLRenderer();
	lines_renderer.setPixelRatio( window.devicePixelRatio );
	lines_renderer.setSize( 512, 512 );
	lines_canvas = lines_renderer.domElement;
	lines_canvas.addEventListener( 'mousemove', onMouseMove );
	document.body.appendChild( lines_canvas );

	controls = new THREE.FlyControls( position_camera, canvas );
	controls.autoForward = true;
	controls.dragToLook = false;
	controls.rollSpeed = Math.PI / 12.;
	controls.movementSpeed = 0.5;
	var gui = new dat.GUI();
	gui.add( config, 'saveImage' ).name( 'Save Image' );
	gui.add( config, 'freeCamera' ).name( 'Free Camera' );
	gui.add( config, 'resolution', [ '256', '512', '800', 'full' ] ).name( 'Resolution' ).onChange( function( value ) {
		if ( value !== 'full' ) {
			canvas.width = value;
			canvas.height = value;
		}
		onWindowResize();
	} );
	stats = new Stats();
	document.body.appendChild( stats.dom );
}

function render( timestamp ) {
	var delta = clock.getDelta();
	stats.begin();
	if ( config.freeCamera ) {
		controls.update( delta );
	} else {
		var _y = 3 * (mouse.y - 0.5);
		var _x = 6 * mouse.x;
		position_camera.position.set( 10. * Math.cos(_x) * Math.cos(_y), 10. * Math.sin(_y), 10. * Math.sin(_x) * Math.cos(_y));
		position_camera.lookAt( new THREE.Vector3( 0.0, 0.0, 0.0 ) );
	}
	// if ( camera.position.y < 0 ) camera.position.y = 0;
	material.uniforms.resolution.value = new THREE.Vector2( canvas.width, canvas.height );
	material.uniforms.cameraPos.value = position_camera.getWorldPosition();
	material.uniforms.cameraDir.value = position_camera.getWorldDirection();
	material.uniforms.time.value = timestamp / 5000.;
	renderer.autoClear = false;
	renderer.render( lines_scene, position_camera );
	renderer.render( scene, dummy_camera );
	// lines_renderer.render( lines_scene, camera );
	stats.end();
	requestAnimationFrame( render );
}


function onMouseMove( e ) {
	mouse.x = e.offsetX / canvas.width;
	mouse.y = e.offsetY / canvas.height;
}
// function onWindowResize( e ) {
// 	if ( config.resolution === 'full' ) {
// 		canvas.width = window.innerWidth;
// 		canvas.height = window.innerHeight;
// 	}
// 	renderer.setSize( canvas.width, canvas.height );
// }
