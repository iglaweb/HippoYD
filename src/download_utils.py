import bz2
import os
from pathlib import Path

import requests

DLIB_LANDMARKS = 'https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2'
CAFFE_RES10_WEIGHTS = 'https://github.com/nhatthai/opencv-face-recognition/raw/master/src/face_detection_model/weights.caffemodel'
CAFFE_RES10_CONFIG = 'https://github.com/nhatthai/opencv-face-recognition/raw/master/src/face_detection_model/deploy.prototxt'
BLAZEFACE_URL = 'https://raw.githubusercontent.com/gouthamvgk/facemesh_coreml_tf/master/keras_models/blazeface_tf.h5'


def download(url, file_name) -> str:
    get_response = requests.get(url, stream=True)
    # file_name = url.split("/")[-1]
    with open(file_name, 'wb') as f:
        for chunk in get_response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return file_name


def download_blazeface(folder) -> str:
    print('Downloading blazeface file...')
    Path(folder).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(folder, "blazeface_tf.h5")
    if os.path.isfile(model_path):  # already exists
        file1 = model_path
    else:
        file1 = download(BLAZEFACE_URL, model_path)
    return file1


def download_caffe(folder) -> (str, str):
    print('Downloading caffe files...')
    Path(folder).mkdir(parents=True, exist_ok=True)
    print('Downloading weights...')
    weights_path = os.path.join(folder, "weights.caffemodel")
    print('Downloading config...')
    config_path = os.path.join(folder, "deploy.prototxt")
    if os.path.isfile(weights_path):  # already exists
        file1 = weights_path
    else:
        file1 = download(CAFFE_RES10_WEIGHTS, weights_path)
    if os.path.isfile(config_path):  # already exists
        file2 = config_path
    else:
        file2 = download(CAFFE_RES10_CONFIG, config_path)
    return file1, file2


def download_and_unpack_dlib_68_landmarks(folder) -> str:
    Path(folder).mkdir(parents=True, exist_ok=True)
    predictor_path_out = os.path.join(folder, "shape_predictor_68_face_landmarks.dat")
    if os.path.exists(predictor_path_out) is True:
        print('Already exists', predictor_path_out)
        return predictor_path_out

    predictor_path = os.path.join(folder, "shape_predictor_68_face_landmarks.dat.bz2")
    if os.path.exists(predictor_path) is False:
        print('Downloading dlib landmarks file...')
        filepath = download(DLIB_LANDMARKS, predictor_path)
    else:
        print(predictor_path + ' already exists')
        filepath = predictor_path

    print('Unzipping file...')
    zipfile = bz2.BZ2File(filepath)  # open the file
    data = zipfile.read()  # get the decompressed data
    newfilepath = filepath[:-4]  # assuming the filepath ends with .bz2
    if os.path.isfile(newfilepath):  # already exists
        return newfilepath

    with open(newfilepath, 'wb') as f:  # write a uncompressed file
        f.write(data)
    return newfilepath
