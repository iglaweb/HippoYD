import os
import sys
from pathlib import Path

# adapt paths for jupyter
from yawn_train.ssd_face_detector import SSDFaceDetector

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import cv2
import dlib
import numpy as np
from imutils import face_utils

# define one constants, for mouth aspect ratio to indicate open mouth
from yawn_train import download_utils, detect_utils
from yawn_train.model_config import MOUTH_AR_THRESH, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT

MOUTH_FOLDER = "./mouth_state_new"
MOUTH_OPENED_FOLDER = f"{MOUTH_FOLDER}/opened"
MOUTH_CLOSED_FOLDER = f"{MOUTH_FOLDER}/closed"

TEMP_FOLDER = "./temp"

# https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset#files
YAWDD_DATASET_FOLDER = "./YawDD dataset"

mouth_open_counter = 0
mouth_close_counter = 0

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


def detect_face_dlib(image) -> list:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    rect_list = []
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        rect_list.append((x, y, x + w, y + h))
    return rect_list


def detect_face_caffe(image) -> list:
    return ssd_face_detector.detect_face(image)


"""
Take mouth ratio only from dlib rect. Use dnn frame for output
"""


def recognize_image(video_path, frame, face_rect_dlib, face_rect_dnn=None):
    (start_x, start_y, endX, endY) = face_rect_dlib
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    face_roi_dlib = frame[start_y:endY, start_x:endX]

    # cv2.imshow('Gray', gray_img)
    # cv2.waitKey(0)

    if face_roi_dlib is None:
        print('Cropped face is None. Skip')
        return

    # determine the facial landmarks for the face region, then
    height_frame, width_frame = face_roi_dlib.shape[:2]

    # https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
    shape = predictor(face_roi_dlib, dlib.rectangle(0, 0, width_frame, height_frame))
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
    video_name = os.path.basename(video_path)
    is_video_no_talking = video_name.endswith('-Normal.avi')
    is_mouth_opened = mouth_mar >= MOUTH_AR_THRESH

    if is_mouth_opened and is_video_no_talking:
        # some videos may contain opened mouth, skip these situations
        return

    if face_rect_dnn is not None:
        (start_x, start_y, endX, endY) = face_rect_dnn
        start_x = max(start_x, 0)
        start_y = max(start_y, 0)
        face_roi_dnn = frame[start_y:endY, start_x:endX]
        target_face_roi = face_roi_dnn
    else:
        target_face_roi = face_roi_dlib

    gray_img = cv2.cvtColor(target_face_roi, cv2.COLOR_BGR2GRAY)
    gray_img = detect_utils.resize_img(gray_img, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)

    if mouth_mar >= MOUTH_AR_THRESH:
        global mouth_open_counter
        mouth_open_counter = mouth_open_counter + 1
        cv2.imwrite(f'{MOUTH_OPENED_FOLDER}/image_{mouth_open_counter}_{mouth_mar}.jpg', gray_img)
    else:
        global mouth_close_counter
        mouth_close_counter = mouth_close_counter + 1
        cv2.imwrite(f'{MOUTH_CLOSED_FOLDER}/image_{mouth_close_counter}_{mouth_mar}.jpg', gray_img)


def process_video(video_path):
    total_img_counter = 0
    face_dlib_counter = 0
    face_caffe_counter = 0
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() is False:
        print('Video is not opened', video_path)
        return
    is_prev_img_dlib = False  # by turns
    while True:
        _, frame = cap.read()
        if frame is None:
            print('No images left. Next video')
            break

        if np.shape(frame) == ():
            print('Empty image. Skip')
            continue

        total_img_counter = total_img_counter + 1
        # reduce img count
        if total_img_counter % 2 != 0:
            continue

        face_list_dlib = detect_face_dlib(frame)
        if len(face_list_dlib) == 0:
            # skip images not recognized by dlib (dlib lndmrks only good when dlib face found)
            continue

        if is_prev_img_dlib is False:
            is_prev_img_dlib = True
            recognize_image(video_path, frame, face_list_dlib[0])
            face_dlib_counter = face_dlib_counter + 1
            continue

        if is_prev_img_dlib is True:
            is_prev_img_dlib = False
            face_list_dnn = detect_face_caffe(frame)
            if len(face_list_dnn) == 0:
                print('Face not found')
                continue

            recognize_image(video_path, frame, face_list_dlib[0], face_list_dnn[0])
            face_caffe_counter = face_caffe_counter + 1

    video_name = os.path.basename(video_path)
    print(
        f"Total images: {total_img_counter}, collected dlib: {face_dlib_counter} images, collected Caffe: {face_caffe_counter} images in video {video_name}")
    cap.release()

    # The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on
    # Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function
    # 'cvDestroyAllWindows'
    try:
        cv2.destroyAllWindows()
    except:
        print('No destroy windows')


def process_videos():
    files_count = 0
    for root, dirs, files in os.walk(YAWDD_DATASET_FOLDER):
        for file in files:
            if file.endswith(".avi"):
                files_count = files_count + 1
                file_name = os.path.join(root, file)
                print(file_name)
                process_video(file_name)

    print(f'Videos processed: {files_count}')
    print(f'Total images: {mouth_open_counter + mouth_close_counter}')
    print(f'Opened mouth images: {mouth_open_counter}')
    print(f'Closed mouth images: {mouth_close_counter}')


if __name__ == '__main__':
    process_videos()
