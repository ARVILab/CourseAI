var container;
var camera, scene, renderer, particles,timelineEl, geometry,wordsSprites, materials = [], curves, parameters, i, h, color, mixer,sprite, size, cameraCtrl;
var mouseX = 0, mouseY = 0;
var clock = new THREE.Clock();
var windowHalfX = window.innerWidth / 2;
var windowHalfY = window.innerHeight / 2;
var yearDiv = document.querySelector('#year')
var loadingManager = new THREE.LoadingManager();
anchor = document.querySelector('#anchor')
paused = false;
loadingManager.onLoad = function () {
	document.getElementById( 'loading' ).style.display = 'none';
    yearDiv.style.display = 'block'
	console.log( 'Done.' );
    createTimeline(1950,2016)
};

showWords = false

loadingManager.onProgress = function ( item, loaded, total ) {
	console.log( loaded + '/' + total, item );
};

yearDiv.addEventListener('click', function(){
    paused = !paused
})

var textureLoader = new THREE.TextureLoader( loadingManager );
textureLoader.load( 'libs/textures/sprites/spark.png', function ( tex ) {
	sprite = tex;
} );


document.querySelector('.gt-timeline').addEventListener('click', function(e) {
    var time = e.offsetX/this.getBoundingClientRect().width
    startTime = new Date - Math.floor(totalTime * time)
})

document.querySelector('.showwords').addEventListener('click', function(){
    showWords = !showWords
})

document.querySelector('#wordsearch').addEventListener('keyup', function(e) {
    if (e.keyCode==13 && this.value.length > 0) {
        searchWord(this.value)
    }
})

var WordCcurve;
CurrentWordID = 0
function searchWord(keyword) {
    var id = words.indexOf(keyword)
    if (id > -1) {
        CurrentWordID = id
        var coords = wordsSprites.children[CurrentWordID]
        camera.lookAt(coords)
        WordCcurve.geometry.vertices = curves[CurrentWordID].getPoints( 500 );

        for (var i =0; i< WordCcurve.geometry.vertices.length; i++ ) {
            WordCcurve.geometry.vertices[i].x *= scaleFactor;
            WordCcurve.geometry.vertices[i].y *= scaleFactor;
            WordCcurve.geometry.vertices[i].z *= scaleFactor;
        }

        WordCcurve.geometry.verticesNeedUpdate = true;
        WordCcurve.visible = true
    } else {
        WordCcurve.visible = false
        this.value = ''
    }
}

function createTimeline(startYear, endYear) {
    timelineEl = document.querySelector('.gt-timeline')
    var yearcount = endYear - startYear
    var pad = Math.floor(timelineEl.clientWidth / yearcount)
    var p = 14
    for (var i=startYear; i < endYear+1; i++) {
        if (pad < 100) {
            if (i%2 == 0) {
                timelineEl.innerHTML += '<div class="horizontal-line leftend" style="left:'+p+'px"><div class="year">'+i+'</div></div>';
            } else {
                timelineEl.innerHTML += '<div class="horizontal-line leftend" style="left:'+p+'px">';
            }
        } else {
            timelineEl.innerHTML += '<div class="horizontal-line leftend" style="left:'+p+'px"><div class="year">'+i+'</div></div>';
        }
        p+=pad;
    }
}

function makeTextSprite( message, parameters ) {
	if (parameters === undefined) parameters = {};

	var fontface = parameters.hasOwnProperty("fontface") ?
		parameters["fontface"] : "Arial";

	var fontsize = parameters.hasOwnProperty("fontsize") ?
		parameters["fontsize"] : 10
	var borderThickness = parameters.hasOwnProperty("borderThickness") ?
		parameters["borderThickness"] : 2;

	var borderColor = parameters.hasOwnProperty("borderColor") ?
		parameters["borderColor"] : {r: 0, g: 0, b: 0, a: 1.0};

	var backgroundColor = parameters.hasOwnProperty("backgroundColor") ?
		parameters["backgroundColor"] : {r: 0, g: 0, b: 0, a: 0.0};

	var canvas = document.createElement('canvas');
	canvas.width = 90
	canvas.height = 20
	var context = canvas.getContext('2d');
	context.font = "Bold " + fontsize + "px " + fontface;

	var metrics = context.measureText(message);
	var textWidth = metrics.width;

	context.fillStyle = "rgba(" + backgroundColor.r + "," + backgroundColor.g + ","
		+ backgroundColor.b + "," + backgroundColor.a + ")";

	context.strokeStyle = "rgba(" + borderColor.r + "," + borderColor.g + ","
		+ borderColor.b + "," + borderColor.a + ")";

	context.fillStyle = "rgba(255, 255, 255, 1.0)";

	context.fillText(message, borderThickness, fontsize + borderThickness);

	var texture = new THREE.Texture(canvas)
	texture.needsUpdate = true;

	var spriteMaterial = new THREE.SpriteMaterial(
		{map: texture, useScreenCoordinates: true});
	var sprite = new THREE.Sprite(spriteMaterial);
	sprite.scale.set(25, 10, 5)
	return sprite
}

var WordParticle
HISTORY = []
timeframes = []
var OBJloader = new THREE.OBJLoader( loadingManager );

function loadYears(year) {
    if (year < 2016)  {
            OBJloader.load( 'language/'+year+'.obj', function ( model ) {
                if (model.children[0].geometry) {
                     HISTORY.push(model.children[0].geometry.vertices);
                     year++
                }

                loadYears(year)
            } );
    } else {
        init();
        animate()
    }
}
loadYears(1950)


function init() {
    container = document.createElement( 'div' );
    document.body.appendChild( container );
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 1, 2000 );
    camera.position.z = 1000;

    cameraCtrl = new THREE.TrackballControls( camera, container );
    cameraCtrl.rotateSpeed = 7.0;
    cameraCtrl.zoomSpeed = 1.2;
    cameraCtrl.panSpeed = 7.8;
    cameraCtrl.noZoom = false;
    cameraCtrl.noPan = false;
    cameraCtrl.staticMoving = true;
    cameraCtrl.dynamicDampingFactor = 6.9;
    cameraCtrl.keys = [ 65, 83, 68 ];
    cameraCtrl.addEventListener( 'change', render );


    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2( 0x000000, 0.0008 );

    geometry = new THREE.Geometry();

    scaleFactor = 200;

    for ( i = 0; i < 7500; i ++ ) {
        var vertex = new THREE.Vector3();
        vertex.x = HISTORY[0][i].x;
        vertex.y = HISTORY[0][i].y;
        vertex.z = HISTORY[0][i].z;
        geometry.vertices.push( vertex );
    }

    curves = []
    for (var i = 0; i < HISTORY[0].length; i++) {
        var points = []
        for (var j=0; j < HISTORY.length; j++) {
            points.push(HISTORY[j][i])
        }
        curves.push(new THREE.CatmullRomCurve3(points))
    }

    framesPerYear = 25;
    for (var i=0; i < framesPerYear*HISTORY.length; i++) {
        timeframes[i] = []
        t = i/(framesPerYear*HISTORY.length);
        for (var k=0 ; k < curves.length; k++) {
            timeframes[i].push(curves[k].getPoint(t))
        }
        if (i%100 ==0) {
            console.log('computed ' + k + ' frames')
        }
    }

    sprite = textureLoader.load( "libs/textures/sprites/spark1.png" );
    material = new THREE.PointsMaterial( { size: 60, map: sprite, blending: THREE.AdditiveBlending, depthTest: false, transparent : true } );
    material.color.setRGB( 46/256, 144/256, 242/256 )

    particles = new THREE.Points( geometry, material );

    scene.add( particles );

    wordsSprites = new THREE.Object3D();

    for (var i =0; i< curves.length; i++) {
		var wordsprite = makeTextSprite(words[i]);
		wordsprite.position.x = curves[i].points[0].x*scaleFactor +2;
		wordsprite.position.y = curves[i].points[0].y*scaleFactor -2;
		wordsprite.position.z = curves[i].points[0].z*scaleFactor +2;
		wordsSprites.add(wordsprite)
	}

    scene.add(wordsSprites)

    var curveGeometry = new THREE.Geometry();
    curveGeometry.vertices = curves[0].getPoints( 500 );

    for (var i =0; i< curveGeometry.vertices.length; i++ ) {
        curveGeometry.vertices[i].x *= scaleFactor;
        curveGeometry.vertices[i].y *= scaleFactor;
        curveGeometry.vertices[i].z *= scaleFactor;
    }

    var material = new THREE.LineBasicMaterial( { color : 0xff0000, linewidth: 2 } );

    // Create the final Object3d to add to the scene
    WordCcurve = new THREE.Line( curveGeometry, material );
    scene.add(WordCcurve)

    var WordMaterial = new THREE.PointsMaterial( { size: 170, map: sprite, blending: THREE.AdditiveBlending, depthTest: false, transparent : true } );
    WordMaterial.color.setRGB( 0, 255/256,0 )
    var wordGeometry = new THREE.Geometry();
    wordGeometry.vertices.push(new THREE.Vector3(0,0,0))
    WordParticle = new THREE.Mesh(wordGeometry, WordMaterial);
    WordParticle.position = geometry.vertices[CurrentWordID];
    WordParticle.visible = true

    scene.add(WordParticle)

    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    container.appendChild( renderer.domElement );

    window.addEventListener( 'resize', onWindowResize, false );
}

function onWindowResize() {

    windowHalfX = window.innerWidth / 2;
    windowHalfY = window.innerHeight / 2;

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

}

function onDocumentMouseMove( event ) {

    mouseX = event.clientX - windowHalfX;
    mouseY = event.clientY - windowHalfY;

}

function onDocumentTouchStart( event ) {

    if ( event.touches.length === 1 ) {

        event.preventDefault();

        mouseX = event.touches[ 0 ].pageX - windowHalfX;
        mouseY = event.touches[ 0 ].pageY - windowHalfY;

    }

}

function onDocumentTouchMove( event ) {

    if ( event.touches.length === 1 ) {

        event.preventDefault();

        mouseX = event.touches[ 0 ].pageX - windowHalfX;
        mouseY = event.touches[ 0 ].pageY - windowHalfY;

    }

}

//

function animate() {

    requestAnimationFrame( animate );
    cameraCtrl.update()
    render();

}
framenumber = 0;
totalTime = 120000
startTime = new Date()
function render() {
    var delta = clock.getDelta();
    if (mixer) {
         //mixer.update( delta );
    }
    if (!paused) {
        particles.visible = true
        wordsSprites.visible = false
        currTime = new Date()
        playtime = ((currTime-startTime)%totalTime)/totalTime;
        var frameid = Math.floor(playtime*timeframes.length);
        if (!showWords) {
            for (var i=0; i < geometry.vertices.length; i ++) {
                var vf = timeframes[frameid][i]
                geometry.vertices[i].x = vf.x * scaleFactor
                geometry.vertices[i].y = vf.y * scaleFactor
                geometry.vertices[i].z = vf.z * scaleFactor
            }
        } else {
            particles.visible = false
            wordsSprites.visible = true
            for (var i=0; i < wordsSprites.children.length; i ++) {
                var vf = timeframes[frameid][i]
                wordsSprites.children[i].position.x = vf.x * scaleFactor
                wordsSprites.children[i].position.y = vf.y * scaleFactor
                wordsSprites.children[i].position.z = vf.z * scaleFactor
            }
        }

        yearDiv.innerHTML = '' + Math.floor(playtime*(2015-1950) + 1950)
        geometry.verticesNeedUpdate = true;
        framenumber++;
        if (timelineEl) {
            document.querySelector('#anchor').style.left = Math.floor(playtime * timelineEl.getBoundingClientRect().width) + 'px'
        }
    }

    if (WordCcurve.visible) {
        WordParticle.position = wordsSprites.children[CurrentWordID].position
        camera.lookAt( wordsSprites.children[CurrentWordID].position );
    } else {
        camera.lookAt( scene.position );
    }
    renderer.render( scene, camera );

}

