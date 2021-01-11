import cv2

from yawn_train.ssd_face_detector import SSDFaceDetector


class VideoFaceDetector(object):

    def __init__(self, filename, face_model):
        self.ssd_face_detector = SSDFaceDetector(face_model)
        self._filename = filename
        self.face_model = face_model
        self.vid = cv2.VideoCapture(filename)
        if self.vid.isOpened() is False:
            raise Exception("Video not opened")

    def start(self, image_reader):
        while True:
            _, frame = self.vid.read()
            face_list = self.ssd_face_detector.detect_face(frame)
            if len(face_list) == 0:
                print('Face list empty')
                cv2.imshow('Image', frame)
                cv2.waitKey(1)
                continue
            image_reader(frame, face_list[0])
        self.vid.release()
