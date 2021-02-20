import os

import cv2
import dlib
from imutils import face_utils

from yawn_train.src import download_utils, inference_utils, detect_utils

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
from ddfa_model.FaceDetection import FaceBoxes_ONNX
from ddfa_model.FaceAlignment3D import TDDFA_ONNX

face_boxes = FaceBoxes_ONNX.FaceBoxes_ONNX()
tddfa = TDDFA_ONNX.TDDFA_ONNX()

TEMP_FOLDER = "./temp"
DETECT_FACE = False
dlib_landmarks_file = download_utils.download_and_unpack_dlib_68_landmarks(TEMP_FOLDER)
# dlib predictor for 68pts
predictor = dlib.shape_predictor(dlib_landmarks_file)
detector = dlib.get_frontal_face_detector()

img = cv2.imread(
    '/yawn_train/out_pred_mouth/0.01_8082_0.6_241_641_dlib.jpg',
    cv2.IMREAD_GRAYSCALE)

if DETECT_FACE:
    rects = inference_utils.detect_face_dlib(detector, img)
    # img2 = cv2.imread(
    #   '/Users/igla/Downloads/mouth_state_new4/closed/95468_0.29_206_244.jpg')
    # bboxes = face_boxes(img2)
    # print(bboxes)
else:
    rects = []
    height_frame, width_frame = img.shape[:2]
    rects.append((0, 0, width_frame, height_frame))

if len(rects) > 0:

    # determine the facial landmarks for the face region, then
    height_frame, width_frame = img.shape[:2]
    face_rect = rects[0]
    (start_x, start_y, end_x, end_y) = face_rect
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    dlib_rect = dlib.rectangle(start_x, start_y, end_x, end_y)

    # https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
    shape = predictor(img, dlib_rect)  # dlib.rectangle(0, 0, width_frame, height_frame))
    shape = face_utils.shape_to_np(shape)

    print(shape)
    print('Predictions size: ' + str(len(shape)))

    mouth = shape[mStart:mEnd]
    mouth_mar = detect_utils.mouth_aspect_ratio(mouth)
    print(mouth_mar)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cnt = 0
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]  # right_eye
    for (x, y) in shape:
        cnt = cnt + 1
        if mStart < cnt <= mEnd:
            # cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(img, str(cnt), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
else:
    print('No face')
cv2.imshow("Dlib landmarks", img)
cv2.waitKey(0)
