existingModels = ['focusRu','subtitlesEn'];
MODELS = {};

updateInterval = 1
if (document.location.search.replace('?model=','') == 'ru') {
	currentModel = 'focusRu'
	updateInterval = 1
} else {
	currentModel = 'subtitlesEn'
	updateInterval = 10
}

uiLoading = true;
currentWordID = 0;
graph = null
firstModelLoaded = false

var createTimeline = function(startYear, endYear) {
    timelineEl = document.querySelector('.yearlines')
    timelineEl.innerHTML = ''
    var yearcount = endYear - startYear
    var pad = Math.floor(timelineEl.clientWidth / yearcount)
    var p = 14;
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
    document.querySelector('.gt-timeline').addEventListener('click', function(e) {
    var time = e.offsetX/this.getBoundingClientRect().width
    graph.startTime = new Date - Math.floor(graph.totalTime * time)
})
}

var searchWord = function(keyword) {
    var id = MODELS[currentModel].words.indexOf(keyword)
    if (id > -1) {
        currentWordID = id
        var coords = wordsSprites.children[currentWordID]
        graph.lookAt(coords)
        WordCcurve.geometry.vertices = curves[currentWordID].getPoints( 500 );

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

new SimpleAjax().get('/models?model='+currentModel, function(modeljson) {
    model = JSON.parse(modeljson);
    MODELS[model.name] = model;

    if (!firstModelLoaded) {
        firstModelLoaded = true
        currentModel = model.name
        graph = new Graph(document.querySelector('#viz'), MODELS[model.name], document.querySelector('.gt-timeline'), function(){
            	document.getElementById( 'loading' ).style.display = 'none';
                document.querySelector('#year').style.display = 'block'
                console.log( 'Done.' )
        })
        createTimeline(model.startYear, model.startYear + model.years.length);
        graph.totalTime =200000;
    }
})


document.querySelector('.showwords').addEventListener('click', function(){
    graph.showWords = !graph.showWords
})

document.querySelector('#wordsearch').addEventListener('keyup', function(e) {
    if (e.keyCode==13 && this.value.length > 0) {
        var keyword = this.value
        var id = graph.languageModel.words.indexOf(keyword)
        if (id > -1) {
            graph.currentWordID = id
            graph.showCurveForWords([id]);
            if (!graph.wordDists[id]) {
                 new SimpleAjax().get('/api?model='+graph.languageModel.name+'&keyword='+keyword, function(jsonData) {
                    yearsDists = JSON.parse(jsonData);
                    graph.wordDists[id] = yearsDists;
                })
            }
            } else {
            graph.wordCurves[0].visible = false
            this.value = ''
        }
    }
})


