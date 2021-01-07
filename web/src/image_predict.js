if(typeof jQuery!=='undefined') {
    console.log('jQuery Loaded');
} else {
    console.log('jQuery not loaded yet');
}

var js_model = 'https://raw.githubusercontent.com/iglaweb/YawnMouthOpenDetect/master/out_epoch_60/tfjs_model_60/model.json'

$("#image-selector").change(function () {
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
	}

	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
});

async function init_models() {
	$('.progress-bar').show();
	try {
		model = await tf.loadLayersModel(js_model);
	} catch(e) {
		console.log("the model could not be loaded")
		console.log(e)
	}
	console.log("the model loaded successfully")
	$('.progress-bar').hide();
}


let model;
$(document).ready(init_models());

predict_image = async function (image) {
	let pre_image = tf.browser.fromPixels(image, 1)
		.resizeNearestNeighbor([100, 100])
		.expandDims()
		.toFloat()
		.div(255.0)
		.reverse(-1);
	console.log(pre_image);
	let predict_result = await model.predict(pre_image).data();
	let probability = predict_result[0];
	return probability;
}

function predict_image_sync(image) {
	console.log('Resolve image');
	console.log(image);
	let pre_image = tf.browser.fromPixels(image, 1)
		.resizeNearestNeighbor([100, 100])
		.expandDims()
		.toFloat()
		.div(255.0)
		.reverse(-1);
	console.log('Image prepared');
	console.log(pre_image);
	let predict_result = model.predict(pre_image).data();
	let probability = predict_result[0];
	return probability;
}

$("#predictBtn").click(async function () {
	let image = $('#selected-image').get(0);
	const start1 = performance.now();
	let probability = await predict_image(image);
	const time1  = Math.round(performance.now() - start1);
    console.log('Face inference time: ' + time1 + ' ms');
	console.log(probability);
	$("#prediction-list").empty();
	$("#prediction-list").append(`<li>probability: ${parseInt(Math.trunc(probability * 100))}%, time elapsed: ${time1} ms</li>`);
});