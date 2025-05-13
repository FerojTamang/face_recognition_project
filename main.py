import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import face_recognition
from PIL import Image, ImageTk
import datetime
import threading
import os

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition App")
        master.geometry("800x700")
        master.configure(bg="#1e1e2f")

        self.video_capture = cv2.VideoCapture(0)
        self.known_face_encodings = []
        self.known_face_names = []
        self.captured_image = None
        self.running_recognition = False

        # Create the "pictures" folder if it doesn't exist
        if not os.path.exists('pictures'):
            os.makedirs('pictures')

        self.video_label = tk.Label(master)
        self.video_label.pack(pady=10)

        self.button_frame = tk.Frame(master, bg="#1e1e2f")
        self.button_frame.pack(pady=20)

        self.capture_button = tk.Button(
            self.button_frame,
            text="📷 Capture Image",
            command=self.capture_image,
            bg="#2ecc71", fg="white",
            font=("Arial", 12, "bold"),
            width=18,
            height=2,
            relief="raised"
        )
        self.capture_button.grid(row=0, column=0, padx=20, pady=10)

        self.test_button = tk.Button(
            self.button_frame,
            text="🧪 Test Recognition",
            command=self.start_recognition,
            state=tk.DISABLED,
            bg="#3498db", fg="white",
            font=("Arial", 12, "bold"),
            width=18,
            height=2,
            relief="raised"
        )
        self.test_button.grid(row=0, column=1, padx=20, pady=10)

        self.update_video_stream()

    def update_video_stream(self):
        ret, frame = self.video_capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        if not self.running_recognition:
            self.master.after(10, self.update_video_stream)

    def capture_image(self):
        ret, frame = self.video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            return

        # Create a unique filename with the current timestamp and save it in the "pictures" folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pictures/captured_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        self.captured_image = frame
        self.test_button.config(state=tk.NORMAL)

        # Prompt the user to enter the person's name
        name = simpledialog.askstring("Input", "Enter the person's name:")
        if name:
            encodings = face_recognition.face_encodings(frame)
            if encodings:
                self.known_face_names.append(name)
                self.known_face_encodings.append(encodings[0])
                messagebox.showinfo("Success", f"Image saved as {filename} and name '{name}' added.")
            else:
                messagebox.showwarning("Warning", "No face detected in the image.")
        else:
            messagebox.showwarning("Input", "No name entered. Image saved without labeling.")

    def start_recognition(self):
        self.running_recognition = True
        threading.Thread(target=self.recognition_loop, daemon=True).start()

    def recognition_loop(self):
        while self.running_recognition:
            ret, frame = self.video_capture.read()
            if not ret:
                break
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

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.running_recognition = False
        self.update_video_stream()

    def on_closing(self):
        self.running_recognition = False
        self.video_capture.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
