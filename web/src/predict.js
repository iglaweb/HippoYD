if(typeof jQuery!=='undefined'){
    console.log('jQuery Loaded');
}
else{
    console.log('jQuery not loaded yet');
}

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


let model;
$(document).ready(async function () {
	$('.progress-bar').show();
	try {
		model = await tf.loadLayersModel('tfjs_model_40/model.json');
	} catch(e) {
		console.log("the model could not be loaded")
		console.log(e)
	}
	console.log("the model loaded successfully")
	$('.progress-bar').hide();
});

$("#predictBtn").click(async function () {
	let image = $('#selected-image').get(0);
	
	let pre_image = tf.browser.fromPixels(image, 1)
		.resizeNearestNeighbor([100, 100])
		.expandDims()
		.toFloat()
		.div(255.0)
		.reverse(-1); 
		console.log(pre_image);
	let predict_result = await model.predict(pre_image).data();
	let probability = predict_result[0];
	console.log(probability);

	$("#prediction-list").empty();
	$("#prediction-list").append(`<li>probability: ${parseInt(Math.trunc(probability * 100))} %</li>`);
});