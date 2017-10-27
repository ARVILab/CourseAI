var Graph = function(renderEl, languageModel, timelineEl, callback) {
    this.languageModel = languageModel;
    this.renderEl = renderEl;
    this.scene = null;
    this.camera = null;
    this.clock = new THREE.Clock();
    this.paused = false;
    this.yearDiv = document.querySelector('#year');
    this.progressDiv = document.querySelector('.percentLoaded')
    var self = this
    this.yearDiv.addEventListener('click', function(){
        self.paused = !self.paused
    })
    this.loadingManager = new THREE.LoadingManager();
    this.anchor = document.querySelector('#anchor')
    this.timelineEl = timelineEl
    this.windowHalfX = window.innerWidth / 2;
    this.windowHalfY = window.innerHeight / 2;
    this.cameraCtrl = null;
    this.scaleFactor = 200;
    this.pointsGeometry = null;
    this.pointsMaterial = null;
    this.currentYear = 0
    this.lastYear = 0
    this.history = []
    this.framesPerYear = Math.round(20*80/this.languageModel.years.length);
    this.sprite = new THREE.TextureLoader().load( "/static/libs/textures/sprites/spark1.png" );
    this.wordCurves = [];
    this.totalTime = 30000;
    this.currentWordID = 0;
    this.lastFrameId = 0;
    this.wordDists = []
    this.callback = callback
    onWindowResize = function(){
        windowHalfX = window.innerWidth / 2;
        windowHalfY = window.innerHeight / 2;

        self.camera.aspect = window.innerWidth / window.innerHeight;
        self.camera.updateProjectionMatrix();
        self.renderer.setSize( window.innerWidth, window.innerHeight );
    }

    window.addEventListener( 'resize', onWindowResize, false );
    this.init()
};

Graph.prototype.init = function(languageModel) {
    uiLoading = true;
    this.timeframes = []
    this.container = this.renderEl;
    this.camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 1, 2000 );
    this.camera.position.z = 1000;

    this.initControls('trackball')

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2( 0x000000, 0.0008 );

    this.pointsGeometry = new THREE.Geometry();
    for (var i=0; i < this.languageModel.words.length; i++) {
        this.pointsGeometry.vertices.push( new THREE.Vector3())
    }

    this.computeHistory(this.languageModel.years)
    this.updateVertices(0)
    this.setColors()

    this.pointsMaterial = new THREE.PointsMaterial({
        size: 60,
        map: this.sprite,
        blending: THREE.AdditiveBlending,
        vertexColors: THREE.VertexColors,
        depthTest: false,
        transparent : true
    } );

    this.points = new THREE.Points( this.pointsGeometry, this.pointsMaterial );
    this.scene.add( this.points );

    this.initWordsSprites();
    this.showCurveForWords([0]);

    this.renderer = new THREE.WebGLRenderer();
    this.renderer.setPixelRatio( window.devicePixelRatio );
    this.renderer.setSize( window.innerWidth, window.innerHeight );
    this.container.appendChild( this.renderer.domElement );
    this.startTime = new Date()
    var self = this;
    var animate = function() {
            requestAnimationFrame( animate );
            self.cameraCtrl.update()
            self.render();
    }
    if (this.callback) {
        this.callback()
    }
    animate()
};

Graph.prototype.computeHistory = function(years) {
    this.history = []
    for (var y = 0; y < years.length;y++){
        this.history.push([]);
        var k = 0
        var year = years[y]
        for (var i=0; i<years[y].length; i+=3){
            var v = new THREE.Vector3();
            v.x = year[i] * this.scaleFactor;
            v.y = year[i+1] * this.scaleFactor;
            v.z = year[i+2] *this.scaleFactor;
            this.history[y].push(v);
        }
    }

    this.curves = [];
    this.wordCount = this.languageModel.words.length;

    for (var i = 0; i < this.wordCount; i++) {
        var points = []
        for (var j=0; j < this.history.length; j++) {
            points.push(this.history[j][i])
        }
        this.curves.push(new THREE.CatmullRomCurve3(points))
    }
    var yearsCount = this.history.length;
    var self = this;
    var computeFrame = function(frameid) {
        if (frameid < self.framesPerYear*yearsCount) {
            self.timeframes.push([]);
            var t = frameid/(self.framesPerYear*yearsCount);
            for (var k=0 ; k < self.curves.length; k++) {
                var p = self.curves[k].getPoint(t);
                self.timeframes[frameid].push([p.x, p.y, p.z])
            }
            if (frameid%100 ==0) {
                console.log('computed ' + frameid + ' frames')
            }
            self.progressDiv.innerHTML = Math.round((frameid/self.framesPerYear*yearsCount)*100) + '%';
            setTimeout(function(){
                var p = frameid+1
                computeFrame(p)
            },2)
        } else {
            console.log('done')
        }
    }
    computeFrame(0)
};

Graph.prototype.showCurveForWords = function(wordsIds) {
    for (var i = 0; i < this.wordCurves.length; i++) {
        this.wordCurves[i].visible = false;
    }

    for (var i = 0; i < wordsIds.length; i++) {
        var curve = this.wordCurves[i]
        if (!curve) {
                var curveGeometry = new THREE.Geometry();
                curveGeometry.vertices = this.curves[wordsIds[i]].getPoints( 500 );
                var curveMaterial = new THREE.LineBasicMaterial( { color: 0xffffff, opacity: 1, linewidth: 2, vertexColors: THREE.VertexColors } );
                var colors = []
                for (var j = 0; j < curveGeometry.vertices.length;j++) {
                    colors[ j ] = new THREE.Color( 0xffffff );
					colors[ j ].setHSL( j / curveGeometry.vertices.length, 1.0, 0.5 );
                }
                curveGeometry.colors = colors;
                curve = new THREE.Line( curveGeometry, curveMaterial );
                this.scene.add(curve)
                curve.visible = true
                this.wordCurves.push(curve)
        } else {
                curve.geometry.vertices = this.curves[wordsIds[i]].getPoints( 500 );
                curve.geometry.verticesNeedUpdate = true
        }
        curve.visible = true
    }
};

Graph.prototype.updateVertices = function(frameId) {
    var frame = this.timeframes[frameId]
    for (var i=0; i < frame.length; i++) {
        this.pointsGeometry.vertices[i].x = frame[i][0]
        this.pointsGeometry.vertices[i].y = frame[i][1]
        this.pointsGeometry.vertices[i].z = frame[i][2]
    }
    this.pointsGeometry.verticesNeedUpdate = true;
}

Graph.prototype.log = function(x) {
    return 1/(1+Math.exp(-x))
}

Graph.prototype.setColors = function(distArray){
    var colors = []
    if (!distArray) {
        for (var i = 0; i < this.wordCount; i++) {
            colors[ i ] = new THREE.Color( 0xffffff );
            colors[ i ].setRGB( 46/256, 144/256, 242/256 )
        }
    }else {
        for (var i = 0; i < this.wordCount; i++) {
            colors[ i ] = new THREE.Color( 0xffffff );
            colors[ i ].setHSL(Math.min(this.log(distArray[i]*7)+0., 1),  Math.min(this.log(distArray[i]*12)+0., 1),0.5 );
        }
        if (this.wordsSprites.visible) {
            for (var i = 0; i < this.wordCount; i++) {
                var sprite = this.wordsSprites.children[i]
                sprite.material.color.setHSL(Math.min(this.log(distArray[i]*7)+0., 1),  Math.min(this.log(distArray[i]*12)+0., 1),0.5 );
                var k = (this.log(distArray[i]*3)*3) + 1
                sprite.scale.set(20*k,5*k,1);
             }
        }
    }
    this.pointsGeometry.colors = colors;
    this.pointsGeometry.colorsNeedUpdate = true
}

Graph.prototype.initWordsSprites = function() {
    this.wordsSprites = new THREE.Object3D();
    var frame = this.timeframes[0]
    for (var i =0; i< this.languageModel.words.length; i++) {
		var wordsprite = this.makeTextSprite(this.languageModel.words[i]);
		wordsprite.position.x = frame[i][0]
		wordsprite.position.y = frame[i][1]
		wordsprite.position.z = frame[i][2]
		this.wordsSprites.add(wordsprite)
	}
    this.scene.add(this.wordsSprites)
}

Graph.prototype.updateWordsSprites = function(frameId) {
    var frame = this.timeframes[frameId]
    for (var i =0; i< this.wordsSprites.children.length; i++) {
        var wordsSprite = this.wordsSprites.children[i]
		wordsSprite.position.x = frame[i][0]
		wordsSprite.position.y = frame[i][1]
		wordsSprite.position.z = frame[i][2]
	}
}

Graph.prototype.initControls = function(type) {
    if (type == 'trackball') {
        this.cameraCtrl = new THREE.TrackballControls( this.camera, this.container );
        this.cameraCtrl.rotateSpeed = 7.0;
        this.cameraCtrl.zoomSpeed = 1.2;
        this.cameraCtrl.panSpeed = 7.8;
        this.cameraCtrl.noZoom = false;
        this.cameraCtrl.noPan = false;
        this.cameraCtrl.staticMoving = true;
        this.cameraCtrl.dynamicDampingFactor = 6.9;
        this.cameraCtrl.keys = [ 65, 83, 68 ];
        this.cameraCtrl.addEventListener( 'change', this.render );
    }
}

Graph.prototype.makeTextSprite = function( message, parameters ) {
	if (parameters === undefined) parameters = {};

	var fontface = parameters.hasOwnProperty("fontface") ?
		parameters["fontface"] : "Arial";

	var fontsize = parameters.hasOwnProperty("fontsize") ?
		parameters["fontsize"] : 16
	var borderThickness = parameters.hasOwnProperty("borderThickness") ?
		parameters["borderThickness"] : 2;

	var borderColor = parameters.hasOwnProperty("borderColor") ?
		parameters["borderColor"] : {r: 0, g: 0, b: 0, a: 1.0};

	var backgroundColor = parameters.hasOwnProperty("backgroundColor") ?
		parameters["backgroundColor"] : {r: 0, g: 0, b: 0, a: 0.0};

	var canvas = document.createElement('canvas');


	canvas.width = 128
	canvas.height = 32
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
	sprite.scale.set(20, 5, 1)
	return sprite
}

Graph.prototype.render = function () {
    var self = this;
    if (!self.startTime) {
        self = graph
    }
    if (!self.paused) {
        var currTime = new Date()
        var playtime = ((currTime-self.startTime)%self.totalTime)/self.totalTime;
        var frameid = Math.floor(playtime*self.timeframes.length);
        self.lastFrameId = frameid;
        if (!self.showWords) {
            self.points.visible = true;
            self.wordsSprites.visible = false;
            self.updateVertices(frameid)
        } else {
            self.points.visible = false;
            self.wordsSprites.visible = true;
            self.updateWordsSprites(frameid)
        }
        self.currentYear = Math.floor(playtime*self.languageModel.years.length + self.languageModel.startYear)
        self.yearDiv.innerHTML = '' + self.currentYear
        document.querySelector('#anchor').style.left = Math.floor(playtime * self.timelineEl.getBoundingClientRect().width) + 'px'
    }

    if (self.currentWordID) {
        if (self.lastYear != self.currentYear) {
            if (self.wordDists[self.currentWordID]) {
                if (self.wordDists[self.currentWordID].length < self.languageModel.years.length) {
                    self.setColors(self.wordDists[self.currentWordID][Math.floor((self.currentYear - self.languageModel.startYear)/updateInterval)])
                } else {
                    self.setColors(self.wordDists[self.currentWordID][(self.currentYear - self.languageModel.startYear)%self.wordDists[self.currentWordID].length])
                }
            }
        }
        var v = new THREE.Vector3()
        v.x = self.timeframes[self.lastFrameId][self.currentWordID][0]
        v.y = self.timeframes[self.lastFrameId][self.currentWordID][1]
        v.z = self.timeframes[self.lastFrameId][self.currentWordID][2]
        self.camera.target =  v
        self.camera.lookAt( v );
    } else {
        self.camera.target = self.scene.position;
        self.camera.lookAt( self.scene.position );
        self.wordCurves[0].visible = false
    }
    self.lastYear = self.currentYear
    self.renderer.render( self.scene, self.camera );
}
