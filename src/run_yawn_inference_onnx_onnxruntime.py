import glob
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as rt
import tensorflow as tf

from yawn_train.src import download_utils, inference_utils
from yawn_train.src.model_config import IMAGE_PAIR_SIZE, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT
from yawn_train.src.video_face_reader import VideoFaceDetector

assert tf.__version__.startswith('2')

print('TensorFlow version: {}'.format(tf.__version__))

"""
Use this to run interference
Helpful links
https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/codelabs/digit_classifier/ml/step2_train_ml_model.ipynb#scrollTo=WFHKkb7gcJei
"""

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# it runs much slower than float version on CPU
# https://github.com/tensorflow/tensorflow/issues/21698#issuecomment-414764709
CONFIDENCE_THRESHOLD = 0.2
VIDEO_FILE = 0  # '/Users/igla/Downloads/Memorable Monologue- Talking in the Third Person.mp4'
TEST_DIR = '../out_test_mouth/'
TEMP_FOLDER = "./temp"

caffe_weights, caffe_config = download_utils.download_caffe(TEMP_FOLDER)
# Reads the network model stored in Caffe framework's format.
face_model = cv2.dnn.readNetFromCaffe(caffe_config, caffe_weights)


onnx_sess = rt.InferenceSession("../out_epoch_30/yawn_model_onnx_30.onnx")
input_name = onnx_sess.get_inputs()[0].name
label_name = onnx_sess.get_outputs()[0].name


def clear_test():
    # clear previous output images
    files = glob.glob(f'{TEST_DIR}*.jpg', recursive=True)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
    Path(TEST_DIR).mkdir(parents=True, exist_ok=True)


def prepare_input_blob(im):
    if im.shape[0] != MAX_IMAGE_WIDTH or im.shape[1] != MAX_IMAGE_HEIGHT:
        im = cv2.resize(im, IMAGE_PAIR_SIZE)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


mouth_open_counter = 0


def image_reader(frame, face):
    (startX, startY, endX, endY) = face
    frame_crop = frame[startY:endY, startX:endX]

    gray_img = prepare_input_blob(frame_crop)
    # im_input_cv.shape
    # im_input_cv.dtype
    # im_input_cv.ravel()[5000:5010]

    time_start = inference_utils.get_timestamp_ms()

    image_frame = gray_img[:, :, np.newaxis]
    image_frame = image_frame / 255.0
    image_frame = np.expand_dims(image_frame, 0).astype(np.float32)

    pred = onnx_sess.run([label_name], {input_name: image_frame})[0]

    pred = np.squeeze(pred)
    pred = round(pred[()], 2)

    time_diff = inference_utils.get_timestamp_ms() - time_start
    print(f'Prediction: {pred:.2f}; time: {time_diff} ms')

    global mouth_open_counter
    is_mouth_opened = True if pred >= CONFIDENCE_THRESHOLD else False
    if is_mouth_opened:
        mouth_open_counter = mouth_open_counter + 1

    cv2.putText(frame, f"Mouth opened {mouth_open_counter}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255),
                2)
    opened_str = "Opened" if is_mouth_opened else "Closed"
    cv2.putText(frame, opened_str, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)


if __name__ == '__main__':
    mouth_open_counter = 0
    clear_test()

    video_face_detector = VideoFaceDetector(VIDEO_FILE, face_model)
    video_face_detector.start_single(image_reader)
    cv2.destroyAllWindows()
