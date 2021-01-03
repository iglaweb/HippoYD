import datetime
import glob
import os
from pathlib import Path

import cv2
import numpy
import numpy as np
import tensorflow as tf

from yawn_train import download_utils, inference_utils
from yawn_train.model_config import IMAGE_PAIR_SIZE

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

CONFIDENCE_THRESHOLD = 0.2
TFLITE_FLOAT_MODEL = './out_epoch_30/yawn_model_float_30.tflite'

# it runs much slower than float version on CPU
# https://github.com/tensorflow/tensorflow/issues/21698#issuecomment-414764709
TFLITE_QUANT_MODEL = './out_epoch_30/yawn_model_quant_30.tflite'
VIDEO_FILE = 0  # '/Users/igla/Downloads/Memorable Monologue- Talking in the Third Person.mp4'
TEST_DIR = './out_close_mouth/'
TEMP_FOLDER = "./temp"
dataset_labels = ['closed', 'opened']

caffe_weights, caffe_config = download_utils.download_caffe(TEMP_FOLDER)
# Reads the network model stored in Caffe framework's format.
face_model = cv2.dnn.readNetFromCaffe(caffe_config, caffe_weights)

interpreter = tf.lite.Interpreter(model_path=TFLITE_FLOAT_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", interpreter.get_output_details()[0]['name'])
print("shape:", interpreter.get_output_details()[0]['shape'])
print("type:", interpreter.get_output_details()[0]['dtype'])

floating_model = input_details[0]['dtype'] == np.float32
print('Floating model is: ' + str(floating_model))

# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

time_elapsed = 0
exec_cnt = 0


def make_interference(image_frame):
    """
        Return True if opened class detected, otherwise False
    """
    # Acquire frame and resize to expected shape [1xHxWx1]
    # add N dim
    image_frame = inference_utils.prepare_image(image_frame, IMAGE_PAIR_SIZE)

    if floating_model:
        # Normalize to [0, 1]
        image_frame = image_frame / 255.0
        images_data = np.expand_dims(image_frame, 0).astype(np.float32)  # or [img_data]
    else:  # 0.00390625 * q
        images_data = np.expand_dims(image_frame, 0).astype(np.uint8)  # or [img_data]

    start = get_timestamp_ms()
    # Inference on input data normalized to [0, 1]
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    global time_elapsed
    global exec_cnt
    diff = get_timestamp_ms() - start
    time_elapsed = time_elapsed + diff
    exec_cnt = exec_cnt + 1
    print(f'Elapsed time: {diff} ms')
    print(f'Avg time: {time_elapsed / exec_cnt}')

    predict_label = np.argmax(output_data)
    score = 100 * output_data[0][predict_label]
    if floating_model is False:
        score = score * 0.00390625

    predicted_confidence = output_data[0][predict_label]
    is_mouth_opened = True if predicted_confidence >= CONFIDENCE_THRESHOLD else False

    # print(np.argmax(output()[0]))
    print("Predicted value for [0, 1] normalization. Label index: {}, confidence: {:2.0f}%"
          .format(predict_label, score))
    return is_mouth_opened


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


def clear_test():
    # clear previous output images¬
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

    clear_test()
    mouth_open_counter = 0
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

            predicted_confidence = make_interference(frame_crop)
            print(predicted_confidence)

            is_mouth_opened = True if predicted_confidence >= CONFIDENCE_THRESHOLD else False
            if is_mouth_opened:
                mouth_open_counter = mouth_open_counter + 1

            cv2.putText(frame, f"Mouth opened {mouth_open_counter}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255),
                        2)
            opened_str = "Opened" if is_mouth_opened else "Closed"
            cv2.putText(frame, opened_str, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Image", frame)
            cv2.waitKey(1)
    vid.release()
    cv2.destroyAllWindows()
