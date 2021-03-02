import glob
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from yawn_train.src import download_utils, inference_utils
from yawn_train.src.model_config import IMAGE_PAIR_SIZE, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, COLOR_CHANNELS
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
CONFIDENCE_THRESHOLD = 0.4
VIDEO_FILE = '/Users/igla/Downloads/YawDD dataset 2/Mirror/Male_mirror Avi Videos/8-MaleGlassesBeard-Yawning.avi'  # '/Users/igla/Downloads/critical_video_yawn.mp4' #'/Users/igla/Downloads/T001yawning.mp4'
TEST_DIR = '../out_test_mouth/'
TEMP_FOLDER = "./temp"
BATCH_IMG_COUNT_PROCESS = 1  # number of images per process
WRITE_VIDEO = True

# Provide trained KERAS model
cv_model = cv2.dnn.readNetFromONNX('/Users/igla/Downloads/out_epoch_80_lite-3/yawn_model_onnx_80.onnx')

caffe_weights, caffe_config = download_utils.download_caffe(TEMP_FOLDER)
# Reads the network model stored in Caffe framework's format.
face_model = cv2.dnn.readNetFromCaffe(caffe_config, caffe_weights)

mouth_open_counter = 0
batch_img_list = []
last_pred_val = 0.0


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
        try:
            im = cv2.resize(im, IMAGE_PAIR_SIZE)
        except:
            print('Exception')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(im,
                                 scalefactor=1 / 255.0,
                                 ddepth=cv2.CV_32F)
    # print(blob.shape)
    blob = blob.reshape(-1, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, COLOR_CHANNELS)
    # print(blob.shape)
    return blob, im


def prepare_input_blob_multiple(images: list):
    img_list = []
    for img in images:
        if img.shape[0] != MAX_IMAGE_WIDTH or img.shape[1] != MAX_IMAGE_HEIGHT:
            img = cv2.resize(img, IMAGE_PAIR_SIZE)
        img_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    blob = cv2.dnn.blobFromImages(img_list,
                                  scalefactor=1 / 255.0,
                                  ddepth=cv2.CV_32F)
    # print(blob.shape)
    blob = blob.reshape(-1, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, COLOR_CHANNELS)
    # print(blob.shape)
    return blob, img_list


video_writer = None


def image_reader(frame, face):
    global video_writer
    if WRITE_VIDEO and video_writer is None:
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter('./output_video.mp4', fourcc, 29.97, (width, height))

    (start_x, startY, endX, endY) = face
    frame_crop = frame[startY:endY, start_x:endX]

    im_input_cv, gray_img = prepare_input_blob(frame_crop)

    # im_input_cv.shape
    # im_input_cv.dtype
    # im_input_cv.ravel()[5000:5010]

    time_start = inference_utils.get_timestamp_ms()
    cv_model.setInput(im_input_cv)

    predictions = cv_model.forward()
    pred = np.squeeze(predictions)
    pred = round(pred[()], 2)
    time_diff = inference_utils.get_timestamp_ms() - time_start
    print(f'Prediction: {pred:.2f}; time: {time_diff} ms')

    global mouth_open_counter
    is_mouth_opened = True if pred >= CONFIDENCE_THRESHOLD else False
    if is_mouth_opened:
        mouth_open_counter = mouth_open_counter + 1

    # add semitransparent overlay
    x, y, w, h = 0, 0, 280, 60
    sub_img = frame[y:y + h, x:x + w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.8, white_rect, 0.5, 1.0)
    # Putting the image back to its position
    frame[y:y + h, x:x + w] = res

    text_color = (0, 150, 0)
    cv2.putText(frame, f"Mouth opened #{mouth_open_counter}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                text_color,
                2)
    opened_str = "State: " + ("opened" if is_mouth_opened else "closed") + ", " + str(pred)
    cv2.putText(frame, opened_str, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    backtorgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    x_offset = 20
    y_offset = 75
    frame[y_offset:y_offset + backtorgb.shape[0], x_offset:x_offset + backtorgb.shape[1]] = backtorgb

    cv2.imshow("Image", frame)
    cv2.waitKey(1)

    if video_writer is not None:
        video_writer.write(frame)


def image_reader_batch(frame, face):
    global batch_img_list
    global last_pred_val

    batch_img_list.append(frame)
    if len(batch_img_list) == BATCH_IMG_COUNT_PROCESS:
        if face is not None:
            last_pred_val = resolve_predictions(batch_img_list, face)
        else:
            last_pred_val = 0.0
        batch_img_list = []  # clear
    else:
        print('Wait next frame, collected:', len(batch_img_list))

    cv2.putText(frame, f"Mouth opened {mouth_open_counter}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255),
                2)
    is_mouth_opened = last_pred_val >= CONFIDENCE_THRESHOLD
    opened_str = "Opened" if is_mouth_opened else "Closed"
    cv2.putText(frame, opened_str, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Prediction: {last_pred_val}", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255),
                2)
    cv2.imshow("Image", frame)
    cv2.waitKey(1)


def resolve_predictions(frames: list, face) -> float:
    time_start = inference_utils.get_timestamp_ms()

    frames_list = []
    for frame in frames:
        (startX, startY, endX, endY) = face
        frame_crop = frame[startY:endY, startX:endX]
        frames_list.append(frame_crop)

    im_input_cvs, gray_imgs = prepare_input_blob_multiple(frames_list)
    cv_model.setInput(im_input_cvs)
    predictions = cv_model.forward()

    pred = np.squeeze(predictions)
    print(pred)

    global mouth_open_counter
    ret_predictions = []
    total = 0.0
    for x in pred:
        x = round(x, 2)
        ret_predictions.append(x)
        total = total + x

        is_mouth_opened = True if x >= CONFIDENCE_THRESHOLD else False
        if is_mouth_opened:
            mouth_open_counter = mouth_open_counter + 1

    avg_open = total / len(ret_predictions)
    time_diff = inference_utils.get_timestamp_ms() - time_start
    print(f'Predictions: {ret_predictions}; time: {time_diff} ms')
    return round(avg_open, 2)


if __name__ == '__main__':
    mouth_open_counter = 0
    clear_test()

    video_face_detector = VideoFaceDetector(VIDEO_FILE, face_model)
    if BATCH_IMG_COUNT_PROCESS == 1:
        video_face_detector.start_single(image_reader)
    else:
        video_face_detector.start_batch(image_reader_batch)
    cv2.destroyAllWindows()

    if video_writer is not None:
        video_writer.release()
