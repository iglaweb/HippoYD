if(typeof jQuery!=='undefined') {
    console.log('jQuery Loaded');
} else {
    console.log('jQuery not loaded yet');
}

let mouthCounter = 0;

var netDet = undefined;
var netDetYawn = undefined;
// whether streaming video from the camera.
let streaming = false;
let isRunning = false;

let outputCanvas = null;
let output = null;
let camera = null; // layout
let cap = null;
let frame = null;
let frameBGR = null;
let frameGray = null;

let enableWebcamButton = null;

//! [Run face detection model]
function detectFaces(img) {

  var blob = cv.blobFromImage(img, 1, {width: 192, height: 144}, [104, 117, 123, 0], false, false);
  netDet.setInput(blob);
  var out = netDet.forward();

  var faces = [];
  for (var i = 0, n = out.data32F.length; i < n; i += 7) {
    var confidence = out.data32F[i + 2];
    var left = out.data32F[i + 3] * img.cols;
    var top = out.data32F[i + 4] * img.rows;
    var right = out.data32F[i + 5] * img.cols;
    var bottom = out.data32F[i + 6] * img.rows;
    left = Math.min(Math.max(0, left), img.cols - 1);
    right = Math.min(Math.max(0, right), img.cols - 1);
    bottom = Math.min(Math.max(0, bottom), img.rows - 1);
    top = Math.min(Math.max(0, top), img.rows - 1);

    if (confidence > 0.5 && left < right && top < bottom) {
      faces.push({x: left, y: top, width: right - left, height: bottom - top})
    }
  }
  blob.delete();
  out.delete();
  return faces;
};
//! [Run face detection model]


//! [Run mouth detection model]
function detectYawnProbability(img) {
  var blob = cv.blobFromImage(img, 1/255.0, {width: 100, height: 100});
  console.log(blob);
  netDetYawn.setInput(blob);
  var out = netDetYawn.forward();
  var preds = [];
  for (var i = 0; i < out.data32F.length; i++) {
    var confidence = out.data32F[i];
    preds.push(confidence)
  }
  blob.delete();
  out.delete();
  return preds;
};
//! [Run mouth detection model]

// MobileNet-SSD
function loadModels(callback) {
  var utils = new Utils('');
  var proto = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy_lowres.prototxt';
  var weights = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel';
  var onnx_yawn = 'https://raw.githubusercontent.com/iglaweb/YawnMouthOpenDetect/master/out_epoch_60/yawn_model_onnx_60.onnx';
  
  var onnx_yawn_name = 'yawn_model_onnx_60.onnx';
  var face_caffe_weights = 'face_detector.caffemodel';
  var face_caffe_config = 'face_detector.prototxt';

  console.log('Load config ' + face_caffe_config);
  utils.createFileFromUrl(face_caffe_config, proto, () => {
    document.getElementById('status').innerHTML = 'Downloading ' + face_caffe_weights;
    utils.createFileFromUrl(face_caffe_weights, weights, () => {
      document.getElementById('status').innerHTML = 'Downloading ' + onnx_yawn_name;
      utils.createFileFromUrl(onnx_yawn_name, onnx_yawn, () => {
        document.getElementById('status').innerHTML = '';
          netDet = cv.readNetFromCaffe(face_caffe_config, face_caffe_weights);
          console.log('Loaded Caffe face model');
          netDetYawn = cv.readNetFromONNX(onnx_yawn_name);
          console.log('Loaded ONNX Mouth model');
          callback();
      });
    });
  });
};

function initUI() {
  enableWebcamButton = document.getElementById('startStopButton');

  // Create a camera object.
  outputCanvas = document.getElementById('canvasOutput');
  output = document.getElementById('output');
  camera = document.createElement("video");
  camera.setAttribute("width", output.width);
  camera.setAttribute("height", output.height);

  stats = new Stats();
  stats.dom.style.position = 'relative';
  stats.dom.style.float = 'right';
  stats.showPanel(0);
  document.getElementById('container').appendChild(stats.dom);
}

function startCamera() {
  if (streaming) return;

  if(camera.srcObject != null) {
    return;
  }

  // If webcam supported, add event listener to button for when user
  // wants to activate it to call enableCam function which we will 
  // define in the next step.
  if (getUserMediaSupported()) {
    //enableWebcamButton.addEventListener('click', enableCam);
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }

  // Get a permission from user to use a camera.
  navigator.mediaDevices.getUserMedia({video: true, audio: false})
    .then(function(stream) {
      camera.srcObject = stream;
      camera.onloadedmetadata = function(e) {
        camera.play();
      };
  }).catch(function(err) {
    console.warn("An error occured! " + err);
  });

  camera.addEventListener("canplay", function(ev){
    if (!streaming) {
      console.log('Started streaming');
      streaming = true;
    }
  }, false);

  //! [Open a camera stream]
  cap = new cv.VideoCapture(camera);
  console.log('Camera image w=' + camera.width + ', h=' + camera.height);
  frame = new cv.Mat(camera.height, camera.width, cv.CV_8UC4);
  frameBGR = new cv.Mat(camera.height, camera.width, cv.CV_8UC3);
  frameGray = new cv.Mat(camera.height, camera.width, cv.CV_8UC1);
  //! [Open a camera stream]
}

//! [Define frames processing]
function captureFrame() {
    if (!streaming) {
        // clean and stop.
        return;
	}
	
    cap.read(frame);  // Read a frame from camera

    stats.begin();
    cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

    const start = performance.now();
    var faces = detectFaces(frameBGR);
    const time  = Math.round(performance.now() - start);
    console.log('Face inference time: ' + time + ' ms');

    faces.forEach(function(rect) {
      cv.rectangle(frame, {x: rect.x, y: rect.y}, {x: rect.x + rect.width, y: rect.y + rect.height}, [0, 255, 0, 255]);

      var faceRoi = frameBGR.roi(rect);
      cv.cvtColor(faceRoi, frameGray, cv.COLOR_BGR2GRAY);
      console.log('Predict image');

      const start = performance.now();
	  var yawn_ret = detectYawnProbability(frameGray);
	  console.log(yawn_ret[0]);
      var yawn_prob = Math.round(yawn_ret[0] * 100) / 100;
      const time  = Math.round(performance.now() - start);
      console.log('Yawn inference time: ' + time + ' ms');
      console.log('Prediction: ' + yawn_prob);

	  if(yawn_prob > 0.2) {
		mouthCounter++;
	  }
	  var mouth_opened_str = yawn_prob > 0.2 ? "Mouth: opened" : "Mouth: closed";
	  var counter_str = "Counter: " + mouthCounter;
	  var mouth_time_str = "Time: " + time + " ms";
	  var topY = 40;

	  cv.putText(frame, "Confidence: " + yawn_prob, {x: 20, y: topY}, cv.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 0, 255]);
      cv.putText(frame, mouth_opened_str, {x: 20, y: topY + 25}, cv.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 0, 255]);
	  cv.putText(frame, mouth_time_str, {x: 20, y: topY + 50}, cv.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 0, 255]);
	  cv.putText(frame, counter_str, {x: 20, y: topY + 75}, cv.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 0, 255]);
	  
	  cv.imshow(outputCanvas, frameGray);
	  
	  faceRoi.delete();
    });

    stats.end();
    cv.imshow(output, frame);
    requestAnimationFrame(captureFrame);
  };
  //! [Define frames processing]

  function stopVideoProcessing() {
    if (frame != null && !frame.isDeleted()) frame.delete();
    if (frameBGR != null && !frameBGR.isDeleted()) frameBGR.delete();
    if (frameGray != null && !frameGray.isDeleted()) frameGray.delete();
  }

  function stopCamera() {
    if (!streaming) return;

    console.log('Stop camera');
    stopVideoProcessing();
    camera.pause();
    camera.srcObject = null;
    
    if(netDetYawn != null) {
      netDetYawn.delete();
    }
    if(netDet != null) {
      netDet.delete();
    }

    streaming = false;
    isRunning = false;
  }

function main() {
  console.log('OpenCV.js is ready');
  if (streaming) return;

  console.log(cv.getBuildInformation());

  // init UI
  initUI();
  // print opencv info
  startCamera();

  enableWebcamButton.onclick = function toggle() {
    if (isRunning) {
      stopCamera();
      enableWebcamButton.innerHTML = 'Start';
    } else {
      function run() {
        isRunning = true;
        startCamera();
        captureFrame();
        enableWebcamButton.innerHTML = 'Stop';
        enableWebcamButton.disabled = false;
      }
      if (netDet == undefined || netDetYawn == undefined) {
        enableWebcamButton.disabled = true;
        loadModels(run);  // Load models and run a pipeline;
      } else {
        run();
      }
    }
  };
  enableWebcamButton.disabled = false;
};