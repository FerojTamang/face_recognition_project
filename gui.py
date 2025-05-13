# gui.py

import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk  # Ensure this import is correct
from face_recognition import FaceRecognition

class FaceRecognitionApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Face Recognition App")
        self.master.geometry("800x700")
        self.master.configure(bg="#1e1e2f")

        self.face_recognition = FaceRecognition()  # Create an instance of the logic class

        self.video_label = tk.Label(self.master)
        self.video_label.pack(pady=10)

        button_frame = tk.Frame(self.master, bg="#1e1e2f")
        button_frame.pack(pady=20)

        capture_button = tk.Button(
            button_frame,
            text="📷 Capture Image",
            command=self._capture_image,
            bg="#2ecc71", fg="white",
            font=("Arial", 12, "bold"),
            width=18,
            height=2,
            relief="raised"
        )
        capture_button.grid(row=0, column=0, padx=20, pady=10)

        self.test_button = tk.Button(
            button_frame,
            text="🧪 Test Recognition",
            command=self._start_recognition,
            state=tk.DISABLED,
            bg="#3498db", fg="white",
            font=("Arial", 12, "bold"),
            width=18,
            height=2,
            relief="raised"
        )
        self.test_button.grid(row=0, column=1, padx=20, pady=10)

    def _capture_image(self):
        frame = self.face_recognition.capture_image()
        if frame is None:
            messagebox.showerror("Error", "Failed to capture image.")
            return

        name = simpledialog.askstring("Input", "Enter the person's name:")
        if name:
            success = self.face_recognition.encode_face(frame, name)
            if success:
                messagebox.showinfo("Success", f"Image captured and name '{name}' added.")
                self.test_button.config(state=tk.NORMAL)
            else:
                messagebox.showwarning("Warning", "No face detected in the image.")
        else:
            messagebox.showwarning("Input", "No name entered. Image saved without labeling.")

    def _start_recognition(self):
        self.face_recognition.start_recognition(self.video_label)

    def on_closing(self):
        self.face_recognition.stop_recognition()
        self.master.destroy()
