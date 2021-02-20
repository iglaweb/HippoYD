import os
import sys

import cv2
import numpy as np

# adapt paths for jupyter
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from yawn_train.src.blazeface_utils import create_letterbox_image, process_detections


class BlazeFaceDetector(object):

    def __init__(self, face_model):
        self.face_model = face_model

    def detect_face(self, orig_frame, draw_rect: bool = False) -> list:
        orig_h, orig_w = orig_frame.shape[0:2]
        frame = create_letterbox_image(orig_frame, 128)
        h, w = frame.shape[0:2]
        input_frame = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(input_frame.astype(np.float32), 0) / 127.5 - 1
        result = self.face_model.predict(input_tensor)[0]
        final_boxes, landmarks_proposals = process_detections(result, (orig_h, orig_w), 5, 0.75, 0.5, pad_ratio=0.5)
        face_list = []
        for bx in final_boxes:
            face_list.append((bx[0], bx[1], bx[2], bx[3]))
        return face_list
