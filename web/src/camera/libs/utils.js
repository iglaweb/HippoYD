function checkFeatures(info, features) {
  var wasmSupported = true, webrtcSupported = true;
  if (features.webrtc) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      webrtcSupported = false;
    }
  }
  if (features.wasm && !window.WebAssembly) {
    wasmSupported = false;
  }

  if (!webrtcSupported || !wasmSupported) {
    var text = "Your web browser doesn't support ";
    var len = text.length;
    if (!webrtcSupported) {
      text += "WebRTC";
    }
    if (!wasmSupported) {
      if (text.length > len) {
        text += " and ";
      }
      text += "WebAssembly"
    }
    text += ".";
    info.innerHTML = text;
    return false;
  } else {
    info.innerHTML = 'Web RTC and WASM supported';
  }
  return true;
}

// Check if webcam access is supported.
function getUserMediaSupported() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

function overrideUrls() {
  //support localhost and real igla.su/mouth-open-js/
  var image_link = document.getElementById("type_image").getAttribute("href");
  var camera_link = document.getElementById("type_camera").getAttribute("href");
  console.log(image_link);
  console.log(camera_link);

  var host = window.location.href;
  real_website = host.includes("igla.su/mouth-open-js");
  console.log(real_website);

  if(real_website) {
    image_link = '/mouth-open-js' + image_link;
    document.getElementById("type_image").href = image_link;
    camera_link = '/mouth-open-js' + camera_link;
    document.getElementById("type_camera").href = camera_link;
  }
}