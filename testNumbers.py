import cv2
import pickle
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3
from Control import WSClient
from config import WS_IP
import websocket

# Load trained model
with open(r'C:\Users\moham\Downloads\Numbers_Project\Models\digits_with_blank_model.p', 'rb') as f:
    model_data = pickle.load(f)
    
    model = model_data['model']
    labels_dict = model_data['labels_dict']

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

engine = pyttsx3.init()

class SignLanguageApp:
    def __init__(self, root):
        self.esp = WSClient()
        self.root = root
        self.root.title("Number Gesture Recognition")
        self.root.geometry("600x700")
        self.root.configure(bg="#1e1e1e")

        # Neon border using canvas
        self.canvas = tk.Canvas(self.root, width=520, height=400, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.create_rectangle(5, 5, 515, 395, outline="#00FFFF", width=3)

        self.video_label = tk.Label(self.canvas, bg="#1e1e1e")
        self.video_label.place(x=10, y=10)

        self.prediction_label = tk.Label(self.root, text="Prediction: ", font=("Arial", 14),
                                         fg="#1E90FF", bg="#1e1e1e")
        self.prediction_label.pack()

        self.sentence_label = tk.Label(self.root, text="", font=("Arial", 16, "bold"),
                                       fg="white", bg="#1e1e1e", wraplength=550)
        self.sentence_label.pack(pady=10)

        # Button frame
        self.button_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.button_frame.pack(pady=15)

        self.reset_button = tk.Button(
            self.button_frame, text="🔁 Reset", font=("Arial", 14, "bold"),
            fg="white", bg="#1E90FF", command=self.reset_sentence,
            width=12, height=2
        )
        self.reset_button.grid(row=0, column=0, padx=10)

        self.speak_button = tk.Button(
            self.button_frame, text="🗣️ Speak", font=("Arial", 14, "bold"),
            fg="white", bg="#1E90FF", command=self.speak_sentence,
            width=12, height=2
        )
        self.speak_button.grid(row=0, column=1, padx=10)

        self.sentence = ""
        self.last_prediction = ""
        self.repeat_count = 0

        self.cap = cv2.VideoCapture(0)

        self.root.bind("<space>", self.add_space)
        self.root.bind("<BackSpace>", self.delete_char)
        self.root.bind("<Return>", self.speak_sentence)
        self.root.bind("<Escape>", lambda event: self.on_close())

        self.update_video()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def add_space(self, event=None):
        self.sentence += " "
        self.prediction_label.config(text="Prediction: ␣ (space)")
        self.sentence_label.config(text=self.sentence)

    def delete_char(self, event=None):
        self.sentence = self.sentence[:-1]
        self.prediction_label.config(text="Prediction: ⌫ (deleted)")
        self.sentence_label.config(text=self.sentence)

    def reset_sentence(self):
        self.sentence = ""
        self.esp.send("*")
        self.prediction_label.config(text="Prediction: ")
        self.sentence_label.config(text=self.sentence)
        engine.say("Sentence reset")
        engine.runAndWait()

    def speak_sentence(self, event=None):
        if self.sentence:
            engine.say(self.sentence)
            engine.runAndWait()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        predicted_label = ""

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_, y_ = [], []
                features = []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    features.append(lm.x - min(x_))
                    features.append(lm.y - min(y_))

                if len(features) == 42:
                    prediction = model.predict([np.asarray(features)])
                    predicted_label = labels_dict[int(prediction[0])]
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if predicted_label == self.last_prediction:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.last_prediction = predicted_label

        if self.repeat_count == 15:
            if predicted_label == "SPACE":
                self.add_space()
                self.esp.send(" ")
            elif predicted_label == "DEL":
                self.delete_char()
                self.esp.send("#")
            elif predicted_label != "nothing":
                self.sentence += predicted_label
                self.prediction_label.config(text=f"Prediction: {predicted_label}")
                self.sentence_label.config(text=self.sentence)
                self.esp.send(predicted_label)  # <-- send number to ESP

            else:
                self.prediction_label.config(text="Prediction: ")

        img = Image.fromarray(rgb)
        img = img.resize((500, 380))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def on_close(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
