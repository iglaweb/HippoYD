import glob
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from yawn_train.src import download_utils, inference_utils
from yawn_train.src.model_config import IMAGE_PAIR_SIZE
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

# Provide trained KERAS model
model = keras.models.load_model('../out_epoch_30/yawn_model_30.h5')

caffe_weights, caffe_config = download_utils.download_caffe(TEMP_FOLDER)
# Reads the network model stored in Caffe framework's format.
face_model = cv2.dnn.readNetFromCaffe(caffe_config, caffe_weights)


def predict_image_data(img_array):
    start = inference_utils.get_timestamp_ms()

    # scale pixel values to [0, 1]
    img_array = img_array.astype(np.float32)
    img_array /= 255.0
    img_array = tf.expand_dims(img_array, axis=0)  # Create batch axis
    model_predictions = model.predict(img_array)
    predicted_confidence = np.max(model_predictions[0])

    diff = inference_utils.get_timestamp_ms() - start
    print(f'Time elapsed {diff} ms')

    is_mouth_opened = True if predicted_confidence >= CONFIDENCE_THRESHOLD else False
    # classes taken from input data
    predicted_label_id = 'opened' if is_mouth_opened else 'closed'
    condition = f">= {CONFIDENCE_THRESHOLD}" if is_mouth_opened else f"< {CONFIDENCE_THRESHOLD}"
    print(
        f"This image is {predicted_confidence:.2f} percent opened mouth,"
        f" {predicted_confidence:.2f} {condition}, class={predicted_label_id})"
    )
    return predicted_confidence


def predict_image_path(input_img):
    loaded_img = keras.preprocessing.image.load_img(
        input_img, target_size=IMAGE_PAIR_SIZE, color_mode="grayscale"
    )
    img_array = keras.preprocessing.image.img_to_array(loaded_img)
    predict_image_data(img_array)


def clear_test():
    # clear previous output images
    files = glob.glob(f'{TEST_DIR}*.jpg', recursive=True)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
    Path(TEST_DIR).mkdir(parents=True, exist_ok=True)


def image_reader(frame, face):
    (startX, startY, endX, endY) = face
    frame_crop = frame[startY:endY, startX:endX]
    img_expanded = inference_utils.prepare_image(frame_crop, IMAGE_PAIR_SIZE)

    prediction = predict_image_data(img_expanded)
    print(prediction)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)


if __name__ == '__main__':
    # test http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ ?
    # https://sites.google.com/view/utarldd/home
    # predict_image_path('./mouth_state/val/closed/image_74637_0.53.jpg')
    clear_test()

    video_face_detector = VideoFaceDetector(VIDEO_FILE, face_model)
    video_face_detector.start_single(image_reader)
    cv2.destroyAllWindows()
