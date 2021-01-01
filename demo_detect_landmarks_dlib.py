import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57
    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar


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
mouth_mar = mouth_aspect_ratio(mouth)
print(mouth_mar)

for (x, y) in shape:
    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
cv2.imshow("Dlib landmarks", img)
cv2.waitKey(0)
