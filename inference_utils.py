import datetime

import cv2
import numpy as np


def prepare_image(image_frame, image_size):
    image_frame = cv2.resize(image_frame, image_size, cv2.INTER_AREA)
    image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    return image_frame[:, :, np.newaxis]  # expand 1 dimension


def get_timestamp_ms():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)


def detect_face_dlib(detector, gray_img) -> list:
    # detect faces in the grayscale image
    rects = detector(gray_img, 1)
    rect_list = []
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        rect_list.append((rect.left(), rect.top(), rect.right(), rect.bottom()))
    return rect_list
