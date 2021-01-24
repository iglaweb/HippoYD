import csv
import os
import sys
from enum import Enum
from pathlib import Path

# adapt paths for jupyter

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from yawn_train.blazeface_detector import BlazeFaceDetector

import cv2
import dlib
import numpy as np
from imutils import face_utils

from yawn_train.ssd_face_detector import SSDFaceDetector

# define one constants, for mouth aspect ratio to indicate open mouth
from yawn_train import download_utils, detect_utils, inference_utils
from yawn_train.model_config import MOUTH_AR_THRESH, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT


class FACE_TYPE(Enum):
    BLAZEFACE = 0
    DLIB = 1
    CAFFE = 2

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    def get_next(self):
        val = self.value
        if self.has_value(val + 1):
            return FACE_TYPE(val + 1)
        return FACE_TYPE(0)


MOUTH_FOLDER = "./mouth_state_new4"
MOUTH_OPENED_FOLDER = os.path.join(MOUTH_FOLDER, 'opened')
MOUTH_CLOSED_FOLDER = os.path.join(MOUTH_FOLDER, 'closed')

TEMP_FOLDER = "./temp"

# https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset#files
YAWDD_DATASET_FOLDER = "./YawDD dataset"
CSV_STATS = 'video_stat.csv'

read_mouth_open_counter = 0
read_mouth_close_counter = 0

saved_mouth_open_counter = 0
saved_mouth_close_counter = 0

PROCESS_EVERY_IMG_OPENED = 1
PROCESS_EVERY_IMG_CLOSED = 4

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

Path(MOUTH_FOLDER).mkdir(parents=True, exist_ok=True)
Path(MOUTH_OPENED_FOLDER).mkdir(parents=True, exist_ok=True)
Path(MOUTH_CLOSED_FOLDER).mkdir(parents=True, exist_ok=True)

dlib_landmarks_file = download_utils.download_and_unpack_dlib_68_landmarks(TEMP_FOLDER)
# dlib predictor for 68pts, mouth
predictor = dlib.shape_predictor(dlib_landmarks_file)
# initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

caffe_weights, caffe_config = download_utils.download_caffe(TEMP_FOLDER)
# Reads the network model stored in Caffe framework's format.
face_model = cv2.dnn.readNetFromCaffe(caffe_config, caffe_weights)
ssd_face_detector = SSDFaceDetector(face_model)

import tensorflow as tf

bf_model = download_utils.download_blazeface(TEMP_FOLDER)
blazeface_tf = tf.keras.models.load_model(bf_model, compile=False)
blazefaceDetector = BlazeFaceDetector(blazeface_tf)
"""
Take mouth ratio only from dlib rect. Use dnn frame for output
"""


def recognize_image(video_id: int, video_path: str, frame, frame_id: int, face_type: FACE_TYPE, face_rect_dlib,
                    face_rect_dnn=None) -> bool:
    (start_x, start_y, end_x, end_y) = face_rect_dlib
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    if start_x >= end_x or start_y >= end_y:
        print('Invalid detection. Skip', face_rect_dlib)
        return False

    face_roi_dlib = frame[start_y:end_y, start_x:end_x]
    if face_roi_dlib is None:
        print('Cropped face is None. Skip')
        return False

    # https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
    shape = predictor(frame, dlib.rectangle(start_x, start_y, end_x, end_y))
    shape = face_utils.shape_to_np(shape)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    # for (x, y) in shape:
    #   cv2.circle(face_roi, (x, y), 1, (0, 0, 255), -1)

    # extract the mouth coordinates, then use the
    # coordinates to compute the mouth aspect ratio
    mouth = shape[mStart:mEnd]
    mouth_mar = detect_utils.mouth_aspect_ratio(mouth)
    # compute the convex hull for the mouth, then
    # visualize the mouth
    # mouthHull = cv2.convexHull(mouth)
    # cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
    # cv2.putText(frame, "MAR: {:.2f}".format(mouth_mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    mouth_mar = round(mouth_mar, 2)

    # not sure if mouth opened or not, so exclude range 0.5 - 0.6
    if MOUTH_AR_THRESH - 0.1 <= mouth_mar < MOUTH_AR_THRESH:
        # print(f'Skip image with mar={mouth_mar}')
        return False

    video_name = os.path.basename(video_path)
    is_video_no_talking = video_name.endswith('-Normal.avi')
    is_mouth_opened = mouth_mar >= MOUTH_AR_THRESH

    if is_mouth_opened and is_video_no_talking:
        # some videos may contain opened mouth, skip these situations
        return False

    prefix = 'dlib'
    target_face_roi = None
    if face_rect_dnn is not None:
        (start_x, start_y, end_x, end_y) = face_rect_dnn
        start_x = max(start_x, 0)
        start_y = max(start_y, 0)
        if start_x < end_x and start_y < end_y:
            face_roi_dnn = frame[start_y:end_y, start_x:end_x]
            target_face_roi = face_roi_dnn
            prefix = face_type.name.lower()

    if target_face_roi is None:
        target_face_roi = face_roi_dlib

    if len(frame.shape) == 2:  # single channel
        gray_img = target_face_roi
    else:
        gray_img = cv2.cvtColor(target_face_roi, cv2.COLOR_BGR2GRAY)
    gray_img = detect_utils.resize_img(gray_img, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)

    if is_mouth_opened:
        global read_mouth_open_counter
        read_mouth_open_counter = read_mouth_open_counter + 1
        # reduce img count
        if read_mouth_open_counter % PROCESS_EVERY_IMG_OPENED != 0:
            return False

        global saved_mouth_open_counter
        saved_mouth_open_counter = saved_mouth_open_counter + 1
        file_name = os.path.join(MOUTH_OPENED_FOLDER,
                                 f'{read_mouth_open_counter}_{mouth_mar}_{video_id}_{frame_id}_{prefix}.jpg')
        cv2.imwrite(file_name, gray_img)
    else:
        global read_mouth_close_counter
        read_mouth_close_counter = read_mouth_close_counter + 1
        # reduce img count
        if read_mouth_close_counter % PROCESS_EVERY_IMG_CLOSED != 0:
            return False

        global saved_mouth_close_counter
        saved_mouth_close_counter = saved_mouth_close_counter + 1
        file_name = os.path.join(MOUTH_CLOSED_FOLDER,
                                 f'{read_mouth_close_counter}_{mouth_mar}_{video_id}_{frame_id}_{prefix}.jpg')
        cv2.imwrite(file_name, gray_img)
    return True


def process_video(video_id, video_path) -> int:
    video_name = os.path.basename(video_path)
    is_video_sunglasses = video_name.rfind('-SunGlasses') != -1
    if is_video_sunglasses:
        # inaccurate landmarks in sunglasses
        print('Video contains sunglasses. Skip', video_name)
        return 0

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() is False:
        print('Video is not opened', video_path)
        return 0
    face_dlib_counter = 0
    face_caffe_counter = 0
    face_blazeface_counter = 0
    frame_id = 0
    face_type = FACE_TYPE.DLIB

    while True:
        ret, frame = cap.read()
        frame_id = frame_id + 1
        if ret is False:
            break
        if frame is None:
            print('No images left in', video_path)
            break

        if np.shape(frame) == ():
            print('Empty image. Skip')
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_list_dlib = inference_utils.detect_face_dlib(detector, gray_frame)
        if len(face_list_dlib) == 0:
            # skip images not recognized by dlib (dlib lndmrks only good when dlib face found)
            continue

        if face_type == FACE_TYPE.DLIB:
            is_processed = recognize_image(video_id, video_path, gray_frame, frame_id, face_type, face_list_dlib[0])
            if is_processed:
                face_dlib_counter = face_dlib_counter + 1
                face_type = face_type.get_next()
            continue

        if face_type == FACE_TYPE.CAFFE:
            face_list_dnn = ssd_face_detector.detect_face(frame)
            if len(face_list_dnn) == 0:
                face_type = face_type.get_next()
                print('Face not found with Caffe DNN')
                continue

            is_processed = recognize_image(video_id, video_path, gray_frame, frame_id, face_type, face_list_dlib[0],
                                           face_list_dnn[0])
            if is_processed:
                face_type = face_type.get_next()
                face_caffe_counter = face_caffe_counter + 1

        if face_type == FACE_TYPE.BLAZEFACE:
            face_list_dnn = blazefaceDetector.detect_face(frame)
            if len(face_list_dnn) == 0:
                face_type = face_type.get_next()
                print('Face not found with Blazeface')
                continue
            is_processed = recognize_image(video_id, video_path, gray_frame, frame_id, face_type, face_list_dlib[0],
                                           face_list_dnn[0])
            if is_processed:
                face_type = face_type.get_next()
                face_blazeface_counter = face_blazeface_counter + 1

    print(
        f"Total images: {face_dlib_counter + face_caffe_counter + face_blazeface_counter}"
        f', dlib: {face_dlib_counter} images'
        f', blazeface: {face_blazeface_counter} images'
        f', caffe: {face_caffe_counter} images in video {video_name}'
    )
    cap.release()

    # The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on
    # Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function
    # 'cvDestroyAllWindows'
    try:
        cv2.destroyAllWindows()
    except:
        print('No destroy windows')

    return face_dlib_counter + face_caffe_counter + face_blazeface_counter


def write_csv_stat(filename, video_count, image_count):
    video_stat_dict_path = os.path.join(MOUTH_FOLDER, CSV_STATS)
    if os.path.isfile(video_stat_dict_path) is False:
        with open(video_stat_dict_path, 'w') as f:
            w = csv.writer(f)
            w.writerow(['Video id', 'File name', 'Image count'])

    # mode 'a' append
    with open(video_stat_dict_path, 'a') as f:
        w = csv.writer(f)
        w.writerow((video_count, filename, image_count))


def process_videos():
    video_count = 0
    for root, dirs, files in os.walk(YAWDD_DATASET_FOLDER):
        for file in files:
            if file.endswith(".avi"):
                video_count = video_count + 1
                file_name = os.path.join(root, file)
                print('Current video', file_name)

                image_count = process_video(video_count, file_name)
                write_csv_stat(file_name, video_count, image_count)

    print(f'Videos processed: {video_count}')
    print(f'Total images: {saved_mouth_open_counter + saved_mouth_close_counter}')
    print(f'Saved opened mouth images: {saved_mouth_open_counter}')
    print(f'Saved closed mouth images: {saved_mouth_close_counter}')


if __name__ == '__main__':
    process_videos()
