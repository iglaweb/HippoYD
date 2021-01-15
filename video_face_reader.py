import threading

import cv2

from yawn_train.ssd_face_detector import SSDFaceDetector


class VideoFaceDetector(object):
    last_face = None
    last_frame = None

    def __init__(self, filename, face_model):
        self.ssd_face_detector = SSDFaceDetector(face_model)
        self.vid = cv2.VideoCapture(filename)
        if self.vid.isOpened() is False:
            raise Exception("Video not opened")

    def start_single(self, image_reader):
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

    def detect_face(self):
        while True:
            try:
                if self.last_frame is not None:
                    face_list = self.ssd_face_detector.detect_face(self.last_frame)
                    if len(face_list) == 0:
                        print('Face not found')
                        continue

                    self.last_face = face_list[0]
                    self.last_frame = None
            except BaseException as e:
                print('{!r}; restarting thread'.format(e))
            else:
                # print('exited normally, bad thread; restarting')
                pass

    def start_batch(self, image_reader):
        # recognize face in background thread
        t = threading.Thread(target=self.detect_face)
        t.daemon = True
        t.start()

        while True:
            ret, frame = self.vid.read()
            if ret is False:
                break
            self.last_frame = frame
            image_reader(frame, self.last_face)
        self.vid.release()
