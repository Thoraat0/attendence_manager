import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock

class FaceRecognitionApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.students = set()
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.csv_filename = f"{self.current_date}.csv"
        self.load_known_faces()

        self.root = BoxLayout(orientation='vertical')
        self.image_widget = Image()
        self.root.add_widget(self.image_widget)

        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return self.root

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_faces_names = []

        for filename in ["amosh.jpg", "rakesh.jpg", "rahul.jpg", "nitin.jpg"]:
            image = face_recognition.load_image_file(f"photos/{filename}")
            encoding = face_recognition.face_encodings(image)[0]
            self.known_face_encodings.append(encoding)

            name = filename.split('.')[0]
            self.known_faces_names.append(name)
            self.students.add(name)

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                if matches.count(True) == 1:  # Check for exact match
                    name = self.known_faces_names[first_match_index]

            self.draw_text_on_frame(frame, name)

        buf = cv2.flip(frame, 0).tobytes()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = image_texture

    def draw_text_on_frame(self, frame, name):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (10, 100)
        font_scale = 1.5
        font_color = (255, 0, 0)
        thickness = 3
        line_type = 2

        cv2.putText(frame, f"{name} Present", bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)

        if name in self.students:
            self.students.remove(name)
            current_time = datetime.now().strftime("%H:%M:%S")
            with open(self.csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([name, current_time])

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    FaceRecognitionApp().run()

