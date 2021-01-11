# HippoYD: Yawn Detector using mouth

## Overview

This is a mouth openness detection model. Full training code, data preparation scripts, and pretrained models are in the repository.

## Requirements

*   Refer to
    [requirements.txt](requirements.txt)
    for dependent libraries that're needed to use the code.

## Image Dataset

The model was trained using video dataset [YawDD][1].

## How to train

1. Convert [YawDD][1] dataset to image folders, 2 classes: `closed` and `opened`
```bash
python convert_dataset_video_to_mouth_img.py
```
2. Split data into 3 datasets: `train`, `validation`, `test`
```bash
python split_data_into_datasets.py
```
3. Train data with:
```bash
python train_yawn.py
```

## Available [pretrained models](out_epoch_60/) and demos

<table>
	<tbody>
		<tr>
         <th>Model</th>
         <th>Epochs</th>
         <th>Example of inference</th>
         <th>Demo</th>
		</tr>
      <tr>
			<td>SavedModel / Keras H5</td>
         <td>60</td>
			<td><a href='run_yawn_inference_tf.py'>run_yawn_inference_tf.py</a></td>
         <td></td>
		</tr>
      <tr>
			<td>TFLite</td>
         <td>60</td>
			<td><a href='run_yawn_inference_tflite.py'>run_yawn_inference_tflite.py</a></td>
         <td></td>
		</tr>
      <tr>
			<td>TensorFlowJS</td>
         <td>60</td>
			<td><a href='image_predict.js'>image_predict.js</a></td>
         <td><a href='https://igla.su/mouth-open-js/'>https://igla.su/mouth-open-js/</a></td>
		</tr>
		<tr>
			<td>ONNX</td>
         <td>60</td>
			<td><a href='run_yawn_inference_onnx.py'>run_yawn_inference_onnx.py</a></td>
         <td></td>
		</tr>
      <tr>
			<td>Frozen pb</td>
         <td>60</td>
			<td><a href='run_yawn_inference_pb.py'>run_yawn_inference_pb.py</a></td>
         <td></td>
		</tr>
	</tbody>
</table>

## Model Inference
<table>
	<tbody>
		<tr>
         <th>Configuration</th>
         <th>Config</th>
         <th>Model</th>
         <th>Time (avg)</th>
         <th>FPS</th>
		</tr>
      <tr>
			<td rowspan="6">Macbook Pro, CPU<br/>2 GHz Quad-Core Intel Core i5</td>
         <td>CPU</td>
			<td>TFLite (Floating)</td>
         <td>5 ms</td>
         <td>250</td>
		</tr>
      <tr>
         <td>CPU</td>
			<td>TFLite (Quantized)</td>
         <td>8 ms</td>
         <td>140</td>
		</tr>
		<tr>
         <td>CPU</td>
			<td>Keras H5 (Floating)</td>
         <td>50 ms</td>
         <td>20</td>
		</tr>
      <tr>
         <td>CPU</td>
			<td>ONNX</td>
         <td>2 ms</td>
         <td>500</td>
		</tr>
      <tr>
         <td>CPU</td>
			<td>Frozen pb</td>
         <td>4 ms</td>
         <td>250</td>
		</tr>
      <tr>
         <td>Wasm (Safari 14.0, Firefox 84)</td>
			<td>TensorFlowJS</td>
         <td>30 ms</td>
         <td>30</td>
		</tr>
	</tbody>
</table>


## Issues

If you find any problems or would like to suggest a feature, please
feel free to file an [issue](https://github.com/iglaweb/YawnMouthOpenDetect/issues)

## License

    Copyright 2021 Igor Lashkov

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

[1]: https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset#files "YawDD dataset"