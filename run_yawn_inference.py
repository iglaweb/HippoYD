import datetime
import glob
import os
from pathlib import Path

import cv2
import dlib
import numpy
import numpy as np
import tensorflow as tf
from imutils import face_utils
from tensorflow import keras

assert tf.__version__.startswith('2')

print('TensorFlow version: {}'.format(tf.__version__))

"""
Use this to run interference
Helpful links
https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/codelabs/digit_classifier/ml/step2_train_ml_model.ipynb#scrollTo=WFHKkb7gcJei
"""


def get_timestamp_ms():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# it runs much slower than float version on CPU
# https://github.com/tensorflow/tensorflow/issues/21698#issuecomment-414764709
IMAGE_DIMENSION = 100
image_size = (IMAGE_DIMENSION, IMAGE_DIMENSION)
CONFIDENCE_THRESHOLD = 0.2
VIDEO_FILE = 0  # '/Users/igla/Downloads/Memorable Monologue- Talking in the Third Person.mp4'
TEST_DIR = './out_test_mouth/'

# KERAS mouth state recognition model
model = keras.models.load_model('./out_epoch_30/yawn_model_30.h5')

# detect dlib landmarks for test
try:
    dlib_predictor = dlib.shape_predictor('../dlib/shape_predictor_68_face_landmarks2.dat')
except RuntimeError:
    dlib_predictor = None

# Reads the network model stored in Caffe framework's format.
face_model = cv2.dnn.readNetFromCaffe('../caffe/deploy2.prototxt', '../caffe/weights.caffemodel')


def detect_face(image):
    # accessing the image.shape tuple and taking the elements
    (h, w) = image.shape[:2]  # get our blob which is our input image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))  # input the blob into the model and get back the detections
    face_model.setInput(blob)
    detections = face_model.forward()
    # Iterate over all of the faces detected and extract their start and end points
    count = 0
    rect_list = []
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            print('Face confidence: ' + str(confidence))
            rect_list.append((startX, startY, endX, endY))
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            count = count + 1  # save the modified image to the Output folder
    return rect_list


def predict_image_data(img_array):
    # scale pixel values to [0, 1]
    img_array = img_array.astype(np.float32)
    img_array /= 255.0
    img_array = tf.expand_dims(img_array, axis=0)  # Create batch axis
    model_predictions = model.predict(img_array)
    predicted_confidence = np.max(model_predictions[0])

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
        input_img, target_size=image_size, color_mode="grayscale"
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


if __name__ == '__main__':
    # test http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ ?
    # https://sites.google.com/view/utarldd/home
    predict_image_path('./mouth_state/val/closed/image_74637_0.53.jpg')

    clear_test()

    vid = cv2.VideoCapture(VIDEO_FILE)
    while True:
        _, frame = vid.read()
        face_list = detect_face(frame)
        if len(face_list) == 0:
            print('Face size empty')
            cv2.imshow('Image', frame)
            cv2.waitKey(1)
            continue

        for face in face_list:
            (startX, startY, endX, endY) = face_list[0]
            frame_crop = frame[startY:endY, startX:endX]

            image_frame = cv2.resize(frame_crop, image_size, cv2.INTER_AREA)
            image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
            img_expanded = image_frame[:, :, np.newaxis]
            prediction = predict_image_data(img_expanded)
            print(prediction)

            if dlib_predictor:
                # determine the facial landmarks for the face region, then
                height_frame, width_frame = frame_crop.shape[:2]
                # https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
                shape = dlib_predictor(frame_crop, dlib.rectangle(0, 0, width_frame, height_frame))
                shape = face_utils.shape_to_np(shape)

            cv2.imshow("Image", frame)
            cv2.waitKey(1)
    vid.release()
    cv2.destroyAllWindows()
