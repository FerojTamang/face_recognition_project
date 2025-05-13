# face_recognition.py

import cv2
import face_recognition
import numpy as np
import datetime
import os
from PIL import Image, ImageTk  # Import Image and ImageTk here

class FaceRecognition:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.known_face_encodings = []
        self.known_face_names = []
        self.running_recognition = False
        self._ensure_picture_folder()

    def _ensure_picture_folder(self):
        if not os.path.exists("pictures"):
            os.makedirs("pictures")

    def capture_image(self):
        frame = self._get_video_frame(bgr=True)
        if frame is None:
            return None

        filename = self._save_image(frame)
        return frame

    def _get_video_frame(self, bgr: bool = False):
        ret, frame = self.video_capture.read()
        if not ret:
            return None
        return frame if bgr else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _save_image(self, frame):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pictures/captured_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename

    def encode_face(self, frame, name):
        encodings = face_recognition.face_encodings(frame)
        if encodings:
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(name)
            return True
        return False

    def start_recognition(self, video_label):
        self.running_recognition = True
        self._recognition_loop(video_label)

    def _recognition_loop(self, video_label):
        while self.running_recognition:
            frame = self._recognize_faces()
            if frame is None:
                break
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame))  # This converts the frame to an image for tkinter
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

        self.running_recognition = False

    def _recognize_faces(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop_recognition(self):
        self.running_recognition = False
        if self.video_capture.isOpened():
            self.video_capture.release()
