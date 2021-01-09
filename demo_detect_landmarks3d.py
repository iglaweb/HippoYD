import cv2
import face_alignment
import torch
from imutils import face_utils
from skimage import io

from yawn_train import detect_utils
from yawn_train.convert_dataset_video_to_mouth_img import MOUTH_AR_THRESH

print(torch.__version__)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

image_path = '/Users/igla/PycharmProjects/DrowsinessClassification/yawn_train/incorrect/0.7_image_77917_0.46.jpg'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=False)
input = io.imread(image_path)
shape = fa.get_landmarks(input)[-1]
print(shape)
print('Predictions size: ' + str(len(shape)))

mouth = shape[mStart:mEnd]
mouth_mar = detect_utils.mouth_aspect_ratio(mouth)
print(mouth_mar)

input = cv2.imread(image_path)
idx = 0
for (x, y) in shape:
    idx = idx + 1
    cv2.circle(input, (x, y), 1, (0, 0, 255), -1)
    # cv2.putText(input, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
cv2.imshow("3D landmarks", input)
cv2.waitKey(0)


def scan_folder(directory):
    import os
    from shutil import copyfile
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            path = os.path.join(directory, filename)
            img = io.imread(path)
            shape = fa.get_landmarks(img)[-1]
            mouth = shape[mStart:mEnd]
            mouth_mar = round(detect_utils.mouth_aspect_ratio(mouth), 2)
            print(mouth_mar)

            filename_only = os.path.splitext(filename)[0]

            img_threshold = filename_only.split("_")
            conf = float(img_threshold[2])
            if conf >= MOUTH_AR_THRESH > mouth_mar or conf < MOUTH_AR_THRESH <= mouth_mar:
                os.makedirs('./incorrect/', exist_ok=True)
                copyfile(path, './incorrect/' + str(mouth_mar) + '_' + os.path.basename(filename))
            continue
        else:
            continue


def filter_out(directory):
    import os
    from shutil import copyfile
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            path = os.path.join(directory, filename)
            img = io.imread(path)
            shape = fa.get_landmarks(img)[-1]
            mouth = shape[mStart:mEnd]
            mouth_mar = round(detect_utils.mouth_aspect_ratio(mouth), 2)
            print(mouth_mar)

            filename_only = os.path.splitext(filename)[0]
            img_threshold = filename_only.split("_")
            conf = float(img_threshold[2])
            if mouth_mar > 1.0 and conf < 0.3:
                print('Filter image by confidence: ' + os.path.basename(filename))
                os.makedirs('./filtered/', exist_ok=True)
                copyfile(path, './filtered/' + str(mouth_mar) + '_' + os.path.basename(filename))
                continue

            # if nose point is most left
            nose_left = shape[29][0] < shape[0][0] and shape[30][0] < shape[0][0] and \
                        shape[29][0] < shape[1][0] and shape[30][0] < shape[1][0]

            nose_right = shape[29][0] > shape[16][0] and shape[30][0] > shape[16][0] and \
                         shape[29][0] > shape[15][0] and shape[30][0] > shape[15][0]

            if nose_left or nose_right:
                print('Filter image: ' + os.path.basename(filename))
                os.makedirs('./filtered/', exist_ok=True)
                copyfile(path, './filtered/' + str(mouth_mar) + '_' + os.path.basename(filename))


print('Filter closed')
filter_out('./mouth_state/closed')
print('Filter opened')
filter_out('./mouth_state/opened')

# print('Scan closed eyes')
# scan_folder('./mouth_state/closed')
# print('Scan opened eyes')
# scan_folder('./mouth_state/opened')
