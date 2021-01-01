# Yawn Detection based on Mouth Openness

# Overview

This model is a mouth openness detection model.

I provide full training code, data preparation scripts, and a pretrained model.


## How to use the pretrained model

To use the pretrained model you will need to download `run_yawn_inference.py` and  
a tensorflow model (`.tflite`, `.h5` or `saved model`).


## Image Dataset
The model was trained using video dataset [YawDD](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset#files).



## Requirements

* tensorflow 2.+ (inference was tested using tensorflow 2.4.0-dev20200810)
* opencv-python


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