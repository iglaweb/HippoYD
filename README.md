# HippoYD: Yawn Detector using mouth

## Overview

This is a mouth openness detection model. Full training code, data preparation scripts, and pretrained models are in the repository.

Open image inference in [Colab](https://colab.research.google.com/drive/10cmolUT0jItiWlaEJ8sr0ZcDigq2l4k4?usp=sharing).<br />

## Demo
![Preview-demo](art/yawn_output_video.gif "Preview demo")<br />

## Requirements

*   Refer to
    [requirements.txt](requirements.txt)
    for dependent libraries that're needed to use the code.

## Image Dataset

The model was trained using two sources:<br />

1. Main source: Video dataset [YawDD][1]<br />
Paper: S. Abtahi, M. Omidyeganeh, S. Shirmohammadi, and B. Hariri, “YawDD: A Yawning Detection Dataset”, Proc. ACM Multimedia Systems, Singapore, March 19 -21 2014.
2. Augmentered by [Kaggle Drowsiness_dataset](https://www.kaggle.com/dheerajperumandla/drowsiness-dataset).

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

## Available [pretrained models](out_epoch_70_pro/) and demos

<table>
	<tbody>
		<tr>
         <th>Model</th>
         <th>Example of inference</th>
         <th>Demo</th>
		</tr>
      <tr>
			<td>SavedModel / Keras H5</td>
			<td><a href='run_yawn_inference_tf_h5.py'>run_yawn_inference_tf_h5.py</a></td>
         <td></td>
		</tr>
      <tr>
			<td>TFLite</td>
			<td><a href='run_yawn_inference_tflite.py'>run_yawn_inference_tflite.py</a></td>
         <td></td>
		</tr>
      <tr>
			<td>TensorFlowJS</td>
			<td><a href='image_predict.js'>image_predict.js</a></td>
         <td><a href='https://igla.su/mouth-open-js/'>https://igla.su/mouth-open-js/</a></td>
		</tr>
		<tr>
			<td>ONNX</td>
			<td><a href='run_yawn_inference_onnx_cv.py'>run_yawn_inference_onnx_cv.py</a><br /><a href='run_yawn_inference_onnx_onnxruntime.py'>run_yawn_inference_onnx_onnxruntime.py</a></td>
         <td><a href='https://igla.su/mouth-open-js/'>https://igla.su/mouth-open-js/</a></td>
		</tr>
      <tr>
			<td>Frozen pb</td>
			<td><a href='run_yawn_inference_tf_pb.py'>run_yawn_inference_tf_pb.py</a></td>
         <td></td>
		</tr>
	</tbody>
</table>

## Model Inference Results
<table>
	<tbody>
		<tr>
         <th>Configuration</th>
         <th>Config</th>
         <th>Model</th>
         <th>Time (avg)</th>
         <th>TFLite ver.</th>
		</tr>
      <tr>
			<td rowspan="6">Macbook Pro, CPU<br/>2 GHz Quad-Core Intel Core i5</td>
         <td>CPU</td>
			<td>TFLite (Floating)</td>
         <td>5 ms</td>
         <td>2.3</td>
		</tr>
      <tr>
         <td>CPU</td>
			<td>TFLite (Quantized)</td>
         <td>8 ms</td>
         <td>2.3</td>
		</tr>
		<tr>
         <td>CPU</td>
			<td>Keras H5 (Floating)</td>
         <td>30 ms</td>
         <td>2.3</td>
		</tr>
      <tr>
         <td>CPU</td>
			<td>ONNX</td>
         <td>2 ms</td>
         <td>2.3</td>
		</tr>
      <tr>
         <td>CPU</td>
			<td>Frozen pb</td>
         <td>4 ms</td>
         <td>2.3</td>
		</tr>
      <tr>
         <td>Wasm (Safari 14.0, Firefox 84)</td>
			<td>TensorFlowJS</td>
         <td>30 ms</td>
         <td>2.3</td>
		</tr>
      <tr>
			<td rowspan="2"><a href='https://www.gsmarena.com/xiaomi_mi_8-9065.php'>Xiaomi MI8</a></td>
         <td>GPU/CPU 3 Threads</td>
			<td>TFLite (Floating)</td>
         <td>4 ms</td>
         <td>2.4</td>
		</tr>
      <tr>
         <td>CPU 3 Threads</td>
			<td>TFLite (Quantized)</td>
         <td>10 ms</td>
         <td>2.4</td>
		</tr>
       <tr>
			<td rowspan="2"><a href='https://www.gsmarena.com/xiaomi_redmi_9-10233.php'>Xiaomi Redmi 9</a></td>
         <td>GPU/CPU 3 Threads</td>
			<td>TFLite (Floating)</td>
         <td>11 ms</td>
         <td>2.4</td>
		</tr>
      <tr>
         <td>CPU 3 Threads</td>
			<td>TFLite (Quantized)</td>
         <td>9 ms</td>
         <td>2.4</td>
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