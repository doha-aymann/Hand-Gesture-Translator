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

# Load trained models for numbers (1 to 10) and alphabet (A-Z)
with open(r'C:\Users\moham\Downloads\Phase1\Models\digits_with_blank_model.p', 'rb') as f:
    model_data_numbers = pickle.load(f)
    model_numbers = model_data_numbers['model']
    labels_dict_numbers = model_data_numbers['labels_dict']

with open(r'C:\Users\moham\Downloads\Phase1\Models\model.p', 'rb') as f:
    model_data_alphabet = pickle.load(f)
    model_alphabet = model_data_alphabet['model']
    labels_dict_alphabet = model_data_alphabet['labels_dict']

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

engine = pyttsx3.init()

class GestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition (Numbers & Alphabets)")
        self.root.geometry("800x600")  # Adjust window size
        self.root.configure(bg="#2C2F36")  # Dark background color

        # Add a nice logo image or background image (for example)
        try:
            bg_image = Image.open(r"C:\Users\moham\Downloads\Phase1\Gemini_Generated_Image_l7kov8l7kov8l7ko.jpeg")
            bg_image = bg_image.resize((800, 600))  # Resize background image to fit window
            bg_image = ImageTk.PhotoImage(bg_image)
            bg_label = tk.Label(self.root, image=bg_image)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Set image to cover the whole window
            bg_label.image = bg_image  # Keep reference to prevent garbage collection
        except Exception as e:
            print(f"Error loading background image: {e}")

        # Title Label with modern font and color
        self.title_label = tk.Label(self.root, text="Gesture Recognition", font=("Helvetica", 24, "bold"),
                                    fg="#00FFFF", bg="#2C2F36")
        self.title_label.pack(pady=30)

        # Buttons for switching forms
        self.number_button = tk.Button(
            self.root, text="Number Prediction", font=("Arial", 16, "bold"),
            fg="white", bg="#1E90FF", command=self.open_number_form,
            width=20, height=2, relief="solid", borderwidth=3
        )
        self.number_button.pack(pady=20)

        self.alphabet_button = tk.Button(
            self.root, text="Alphabet Prediction", font=("Arial", 16, "bold"),
            fg="white", bg="#1E90FF", command=self.open_alphabet_form,
            width=20, height=2, relief="solid", borderwidth=3
        )
        self.alphabet_button.pack(pady=20)

    def open_number_form(self):
        number_form = PredictionForm(self.root, "numbers")

    def open_alphabet_form(self):
        alphabet_form = PredictionForm(self.root, "alphabets")

class PredictionForm:
    def __init__(self, root, model_choice):
        self.model_choice = model_choice
        self.esp = WSClient()
        self.window = tk.Toplevel(root)
        self.window.title(f"{model_choice.capitalize()} Gesture Recognition")
        self.window.geometry("800x600")
        self.window.configure(bg="#1e1e1e")

        # Canvas for video feed
        self.canvas = tk.Canvas(self.window, width=520, height=400, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.create_rectangle(5, 5, 515, 395, outline="#00FFFF", width=3)

        self.video_label = tk.Label(self.canvas, bg="#1e1e1e")
        self.video_label.place(x=10, y=10)

        # Prediction label with modern font style
        self.prediction_label = tk.Label(self.window, text="Prediction: ", font=("Arial", 16, "bold"),
                                         fg="#1E90FF", bg="#1e1e1e")
        self.prediction_label.pack()

        self.sentence_label = tk.Label(self.window, text="", font=("Arial", 18, "bold"),
                                       fg="white", bg="#1e1e1e", wraplength=550)
        self.sentence_label.pack(pady=10)

        # Button frame with a sleek design
        self.button_frame = tk.Frame(self.window, bg="#1e1e1e")
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

        # Initialize model choice
        self.model_choice = model_choice
        self.sentence = ""
        self.last_prediction = ""
        self.repeat_count = 0
        self.cap = cv2.VideoCapture(0)

        # Bind keyboard events
        self.window.bind("<space>", self.add_space)
        self.window.bind("<BackSpace>", self.delete_char)
        self.window.bind("<Return>", self.speak_sentence)
        self.window.bind("<Escape>", lambda event: self.on_close())

        self.update_video()

    def add_space(self, event=None):
        self.sentence += " "
        self.prediction_label.config(text="Prediction: ␣ (space)")
        self.sentence_label.config(text=self.sentence)

    def delete_char(self, event=None):
        self.sentence = self.sentence[:-1]
        self.esp.send("#")
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
                    # Use the appropriate model based on the selected option
                    if self.model_choice == 'numbers':
                        prediction = model_numbers.predict([np.asarray(features)])
                        predicted_label = labels_dict_numbers[int(prediction[0])]
                    else:  # Alphabet model
                        prediction = model_alphabet.predict([np.asarray(features)])
                        predicted_label = labels_dict_alphabet[int(prediction[0])]

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
                self.esp.send(predicted_label)  # <-- send the gesture to ESP

            else:
                self.prediction_label.config(text="Prediction: ")

        img = Image.fromarray(rgb)
        img = img.resize((500, 380))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.window.after(10, self.update_video)

    def on_close(self):
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureRecognitionApp(root)
    root.mainloop()
