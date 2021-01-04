import cv2

import numpy as np


class VideoFaceDetector(object):

    def __init__(self, filename, face_model):
        self._filename = filename
        self.face_model = face_model
        self.vid = cv2.VideoCapture(filename)
        if self.vid.isOpened() is False:
            raise Exception("Video not opened")

    def detect_face(self, image):
        # accessing the image.shape tuple and taking the elements
        (h, w) = image.shape[:2]  # get our blob which is our input image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))  # input the blob into the model and get back the detections
        self.face_model.setInput(blob)
        detections = self.face_model.forward()
        # Iterate over all of the faces detected and extract their start and end points
        count = 0
        rect_list = []
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                # print('Face confidence: ' + str(confidence))
                rect_list.append((startX, startY, endX, endY))
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                count = count + 1  # save the modified image to the Output folder
        return rect_list

    def start(self, image_reader):
        while True:
            _, frame = self.vid.read()
            face_list = self.detect_face(frame)
            if len(face_list) == 0:
                print('Face list empty')
                cv2.imshow('Image', frame)
                cv2.waitKey(1)
                continue
            image_reader(frame, face_list[0])
        self.vid.release()
