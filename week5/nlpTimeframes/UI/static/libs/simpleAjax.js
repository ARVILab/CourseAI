"use strict";
var SimpleAjax = function () {
};

SimpleAjax.prototype.get = function (url, callback, error) {
    var http = new XMLHttpRequest();
    http.open("GET", url, true);
    http.onreadystatechange = function () {
        if (http.readyState === 4) {
            if (http.status === 200) {
                var result = "";
                if (http.responseText) {
                    result = http.responseText;
                }
                if (callback) {
                    callback(result);
                }
            } else {
                if (error) {
                    error(http);
                }
            }
        }
    };
    http.send(null);
    return http;
};

SimpleAjax.prototype.loadBinary = function (url, callback, error) {
    var http = new XMLHttpRequest();
    http.open("GET", url, true);
    http.responseType = 'arraybuffer';
    http.onreadystatechange = function () {
        if (http.readyState === 4) {
            if (http.status === 200) {
                if (callback) {
                    callback(http.response);
                }
            }
            if (http.status === 304) {
                if (callback) {
                    callback(http.response);
                }
            }
            if (http.status === 400) {
                if (error) {
                    error();
                }
            }
        }
    };
    http.onerror = function () {
        error();
    };
    http.send(null);
};

SimpleAjax.prototype.post = function (url, data, callback, error) {
    var http = new XMLHttpRequest();
    http.open("POST", url, true);
    http.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    http.onreadystatechange = function () {
        if (http.readyState == 4) {
            if (http.status == 200) {
                var result = "";
                if (http.responseText)
                    result = http.responseText;
                if (callback)
                    callback(result);
            } else {
                if (error)
                    error(http);
                if (console)
                    console.error("SimpleAjax: POST failed", http);
            }
        }
    };
    http.send(data);
};

SimpleAjax.prototype.put = function (url, data, callback, error) {
    var http = new XMLHttpRequest();
    http.open("PUT", url, true);
    http.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    http.onreadystatechange = function () {
        if (http.readyState == 4) {
            if (http.status == 200) {
                var result = "";
                if (http.responseText)
                    result = http.responseText;
                if (callback)
                    callback(result);
            } else {
                if (error)
                    error(http);
                if (console)
                    console.error("SimpleAjax: PUT failed", http);
            }
        }
    };
    http.send(data);
};