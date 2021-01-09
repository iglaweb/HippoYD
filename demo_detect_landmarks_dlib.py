import cv2
import dlib
from imutils import face_utils

from yawn_train import detect_utils

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# dlib predictor for 68pts, mouth
predictor = dlib.shape_predictor('../dlib/shape_predictor_68_face_landmarks.dat')
img = cv2.imread(
    '/Users/igla/PycharmProjects/DrowsinessClassification/yawn_train/mouth_state/opened/image_5842_0.7.jpg')

# determine the facial landmarks for the face region, then
height_frame, width_frame = img.shape[:2]

# https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
shape = predictor(img, dlib.rectangle(0, 0, width_frame, height_frame))
shape = face_utils.shape_to_np(shape)

print(shape)
print('Predictions size: ' + str(len(shape)))

mouth = shape[mStart:mEnd]
mouth_mar = detect_utils.mouth_aspect_ratio(mouth)
print(mouth_mar)

for (x, y) in shape:
    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
cv2.imshow("Dlib landmarks", img)
cv2.waitKey(0)
