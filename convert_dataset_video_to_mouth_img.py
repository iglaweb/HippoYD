import os
from pathlib import Path

import cv2
import dlib
import numpy
from imutils import face_utils
from pandas import np
from scipy.spatial import distance as dist

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.6
MOUTH_FOLDER = "mouth_state"
MOUTH_OPENED_FOLDER = f"{MOUTH_FOLDER}/opened"
MOUTH_CLOSED_FOLDER = f"{MOUTH_FOLDER}/closed"

# https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset#files
YAWNDD_DATASET_FOLDER = "/Users/igla/Downloads/YawDD dataset"

MAX_IMAGE_HEIGHT = 100
MAX_IMAGE_WIDTH = 100

mouth_open_counter = 0
mouth_close_counter = 0

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

Path(MOUTH_FOLDER).mkdir(parents=True, exist_ok=True)
Path(MOUTH_OPENED_FOLDER).mkdir(parents=True, exist_ok=True)
Path(MOUTH_CLOSED_FOLDER).mkdir(parents=True, exist_ok=True)

# dlib predictor for 68pts, mouth
predictor = dlib.shape_predictor('../dlib/shape_predictor_68_face_landmarks.dat')
# Reads the network model stored in Caffe framework's format.
face_model = cv2.dnn.readNetFromCaffe('../caffe/deploy.prototxt', '../caffe/weights.caffemodel')


def detect_face(image):
    # accessing the image.shape tuple and taking the elements
    (h, w) = image.shape[:2]  # get our blob which is our input image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))  # input the blob into the model and get back the detections
    face_model.setInput(blob)
    detections = face_model.forward()
    # Iterate over all of the faces detected and extract their start and end points
    rect_list = []
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]
        if confidence >= 0.4:
            # print('Face confidence: ' + str(confidence))
            rect_list.append((startX, startY, endX, endY))
    return rect_list


def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    # print(f'A: {A}, B: {B}, C: {C}')

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar


def resize_img(frame_crop, max_width, max_height):
    height, width = frame_crop.shape[:2]
    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        frame_crop = cv2.resize(frame_crop, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame_crop


def recognize_image(frame, face_rect):
    (start_x, start_y, endX, endY) = face_rect
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    face_roi = frame[start_y:endY, start_x:endX]

    # cv2.imshow('Gray', gray_img)
    # cv2.waitKey(0)

    if face_roi is None:
        print('Cropped face is None. Skip')
        return

    # determine the facial landmarks for the face region, then
    height_frame, width_frame = face_roi.shape[:2]

    # https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
    shape = predictor(face_roi, dlib.rectangle(0, 0, width_frame, height_frame))
    shape = face_utils.shape_to_np(shape)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    # for (x, y) in shape:
    #   cv2.circle(face_roi, (x, y), 1, (0, 0, 255), -1)

    # extract the mouth coordinates, then use the
    # coordinates to compute the mouth aspect ratio
    mouth = shape[mStart:mEnd]

    mouth_mar = mouth_aspect_ratio(mouth)
    # compute the convex hull for the mouth, then
    # visualize the mouth
    # mouthHull = cv2.convexHull(mouth)
    # cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
    # cv2.putText(frame, "MAR: {:.2f}".format(mouth_mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    mouth_mar = round(mouth_mar, 2)

    gray_img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray_img = resize_img(gray_img, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)

    if mouth_mar >= MOUTH_AR_THRESH:
        global mouth_open_counter
        mouth_open_counter = mouth_open_counter + 1
        cv2.imwrite(f'{MOUTH_OPENED_FOLDER}/image_{mouth_open_counter}_{mouth_mar}.jpg', gray_img)
    else:
        global mouth_close_counter
        mouth_close_counter = mouth_close_counter + 1
        cv2.imwrite(f'{MOUTH_CLOSED_FOLDER}/image_{mouth_close_counter}_{mouth_mar}.jpg', gray_img)


def process_video(video):
    total_img_counter = 0
    face_img_counter = 0
    cap = cv2.VideoCapture(video)
    while True:
        _, frame = cap.read()
        if frame is None:
            print('No images left. Exit')
            break

        if np.shape(frame) == ():
            print('Empty image. Skip')
            continue

        total_img_counter = total_img_counter + 1
        # reduce size
        if total_img_counter % 2 != 0:
            continue

        face_list = detect_face(frame)
        if len(face_list) == 0:
            print('Face not found')
            # cv2.imshow('Image', frame)
            # cv2.waitKey(1)
            continue

        face_img_counter = face_img_counter + 1
        recognize_image(frame, face_list[0])

    video_name = os.path.basename(video)
    print(f"Total images: {total_img_counter}, collected: {face_img_counter} images in video {video_name}")
    cap.release()
    cv2.destroyAllWindows()


def process_videos():
    files_count = 0
    for root, dirs, files in os.walk(YAWNDD_DATASET_FOLDER):
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
