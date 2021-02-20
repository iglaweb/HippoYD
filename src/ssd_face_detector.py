import cv2

import numpy as np


class SSDFaceDetector(object):

    def __init__(self, face_model):
        self.face_model = face_model

    def detect_face(self, image, draw_rect: bool = False) -> list:
        # accessing the image.shape tuple and taking the elements
        (h, w) = image.shape[:2]  # get our blob which is our input image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))  # input the blob into the model and get back the detections
        self.face_model.setInput(blob)
        detections = self.face_model.forward()
        # Iterate over all of the faces detected and extract their start and end points
        rect_list = []
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(startX, 0)
            startY = max(startY, 0)
            confidence = detections[0, 0, i, 2]
            if confidence >= 0.4:
                # print('Face confidence: ' + str(confidence))
                rect_list.append((startX, startY, endX, endY))
                if draw_rect:
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        return rect_list
