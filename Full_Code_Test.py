import cv2
import pickle
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3
from gtts import gTTS
import pygame
import time
import os
import joblib
from Control import WSClient
from config import WS_IP
import warnings
warnings.filterwarnings("ignore")

# Load models and mappings
with open(r'C:\Users\moham\Downloads\Phase1\Models\digits_model.p', 'rb') as f:
    model_data_numbers = pickle.load(f)
    model_numbers = model_data_numbers['model']
    labels_dict_numbers = model_data_numbers['labels_dict']
    labels_dict_numbers[1] = "0"
    #print(f"number labels{labels_dict_numbers}")

with open(r'C:\Users\moham\Downloads\Phase1\Models\model.p', 'rb') as f:
    model_data_alphabet = pickle.load(f)
    model_alphabet = model_data_alphabet['model']
    labels_dict_alphabet = model_data_alphabet['labels_dict']

with open(r'C:\Users\moham\Downloads\Phase1\Models\arabic_model.pkl', 'rb') as f:
    model_data_arabic = pickle.load(f)
    model_arabic = model_data_arabic['model']
    labels_dict_arabic = model_data_arabic['label_dict']

# Load English Words model
with open(r'C:\Users\moham\Downloads\English_Words\svm_model_with_labels.pkl', 'rb') as f:
    saved_data_english = joblib.load(f)
    model_english_words = saved_data_english['model']
    label_classes_english = saved_data_english['label_classes']
    labels_dict_english = {i: label for i, label in enumerate(label_classes_english)}
    print(f"english words :{labels_dict_english}")
arabic_map = {
    "Alef": "أ", "Beh": "ب", "Teh": "ت", "Theh": "ث", "Jeem": "ج",
    "Hah": "ح", "Khah": "خ", "Dal": "د", "thal": "ذ", "Reh": "ر",
    "Zain": "ز", "Seen": "س", "Sheen": "ش", "Sad": "ص", "Dad": "ض",
    "Tah": "ط", "Zah": "ظ", "Ain": "ع", "Ghain": "غ", "Feh": "ف",
    "Qaf": "ق", "Kaf": "ك", "Lam": "ل", "Meem": "م", "Noon": "ن",
    "Laa": "لا", "Al": "ال", "Teh_Marbuta": "ة", "Heh": "ه",
    "Waw": "و", "Yeh": "ي", "space": " ", "del": "⌫", "nothing": ""
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

engine = pyttsx3.init()


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Translator")
        self.root.attributes('-fullscreen', True)
        
        # Initialize components
        self.cap = cv2.VideoCapture(0)
        self.model_choice = "alphabets"
        self.esp = WSClient()
        self.sentence = ""
        self.last_prediction = ""
        self.repeat_count = 0
        self.is_camera_on = True

        # Background Image
        try:
            self.bg_image = Image.open(r"C:\\Users\\moham\\Downloads\\Phase1\\Gemini_Generated_Image_l7kov8l7kov8l7ko.jpeg")
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            self.bg_label = tk.Label(root, image=self.bg_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"Error loading background image: {e}")

        # Top Control Frame
        top_control_frame = tk.Frame(root, bg='#2C2F36')
        top_control_frame.place(relx=0.02, rely=0.02, anchor='nw')

        # Camera Control Button
        self.camera_btn = tk.Button(top_control_frame, text="📷 Camera: ON", font=("Arial", 12, "bold"),
                                  fg="white", bg="#1E90FF", command=self.toggle_camera,height=2,
                                  width=12, relief="flat")
        self.camera_btn.pack(side=tk.LEFT, padx=5)

        # Window Control Buttons (Top-Right)
        control_frame = tk.Frame(root, bg='#2C2F36')
        control_frame.place(relx=0.98, rely=0.02, anchor='ne')

        # Minimize Button
        tk.Button(control_frame, text="─", font=("Arial", 14, "bold"),
                fg="white", bg="#505050", command=root.iconify,
                width=2, height=1, relief="flat").pack(side=tk.LEFT, padx=2)

        # Maximize/Restore Button
        self.maximize_btn = tk.Button(control_frame, text="◻", font=("Arial", 14, "bold"),
                                    fg="white", bg="#505050", command=self.toggle_fullscreen,
                                    width=2, height=1, relief="flat")
        self.maximize_btn.pack(side=tk.LEFT, padx=2)

        # Close Button
        tk.Button(control_frame, text="✕", font=("Arial", 14, "bold"),
                fg="white", bg="#ff4444", command=self.on_close,
                width=2, height=1, relief="flat").pack(side=tk.LEFT, padx=2)

        # Video Feed
        self.video_frame = tk.Frame(root, bg="#2C2F36")
        self.video_frame.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
        
        self.canvas = tk.Canvas(self.video_frame, width=800, height=600, 
                              bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_rectangle(5, 5, 795, 595, outline="#00FFFF", width=3)
        self.video_label = tk.Label(self.canvas, bg="#1e1e1e")
        self.video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Model Selection Radio Buttons
        self.model_var = tk.StringVar(value="alphabets")
        self.radio_frame = tk.Frame(root, bg="#2C2F36")
        self.radio_frame.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

        self.radios = [
            ("English Alphabet", "alphabets"),
            ("Numbers", "numbers"),
            ("Arabic Letters", "arabic"),
            ("English Words", "english_words")
        ]

        for text, value in self.radios:
            rb = tk.Radiobutton(self.radio_frame, text=text, variable=self.model_var,
                    value=value, font=("Arial", 14, "bold"),
                    bg="#2C2F36", fg="white", selectcolor="#1E90FF",
                    activebackground="#1E90FF",indicatoron=0,width=12,padx=10,pady=5,
                    command=self.update_model)
            rb.pack(side=tk.LEFT, padx=20, ipadx=15, ipady=20)

        # Prediction Label
        self.prediction_label = tk.Label(root, text="Prediction: ", 
                                       font=("Arial", 18, "bold"), fg="#1E90FF", bg="#2C2F36")
        self.prediction_label.place(relx=0.5, rely=0.78, anchor=tk.CENTER)

        # Sentence Display
        self.sentence_label = tk.Label(root, text="", font=("Arial", 20, "bold"),
                                     fg="white", bg="#2C2F36", wraplength=1200)
        self.sentence_label.place(relx=0.5, rely=0.84, anchor=tk.CENTER)

        # Control Buttons
        self.button_frame = tk.Frame(root, bg="#2C2F36")
        self.button_frame.place(relx=0.5, rely=0.92, anchor=tk.CENTER)

        self.reset_btn = tk.Button(self.button_frame, text="🔁 Reset", font=("Arial", 14, "bold"),
                                fg="white", bg="#1E90FF", command=self.reset_sentence,
                                width=15, height=2)
        self.reset_btn.grid(row=0, column=0, padx=20, pady=5)

        self.speak_btn = tk.Button(self.button_frame, text="🗣️ Speak", font=("Arial", 14, "bold"),
                                fg="white", bg="#1E90FF", command=self.speak_sentence,
                                width=15, height=2)
        self.speak_btn.grid(row=0, column=1, padx=20, pady=5)

        # Key bindings
        self.root.bind("<Escape>", lambda event: self.on_close())
        self.root.bind("<space>", self.add_space)
        self.root.bind("<BackSpace>", self.delete_char)
        self.root.bind("<Return>", self.speak_sentence)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_video()

    def toggle_camera(self):
        self.is_camera_on = not self.is_camera_on
        state = "ON" if self.is_camera_on else "OFF"
        color = "#1E90FF" if self.is_camera_on else "#505050"
        self.camera_btn.config(text=f"📷 Camera: {state}", bg=color)
        self.last_prediction = ""
        self.repeat_count = 0

    def toggle_fullscreen(self):
        current_state = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current_state)
        self.maximize_btn.config(text="❒" if current_state else "◻")

    def on_close(self):
        """Clean up resources and close application"""
        self.cap.release()
        self.root.destroy()

    def update_model(self):
        self.model_choice = self.model_var.get()
        if self.model_choice == "arabic":
            self.reset_btn.config(text="🔁 إعادة تعيين")
            self.speak_btn.config(text="🗣️ نطق الجملة")
            self.prediction_label.config(text="الحرف المتوقع: ")
        else:
            self.reset_btn.config(text="🔁 Reset")
            self.speak_btn.config(text="🗣️ Speak")
            self.prediction_label.config(text="Prediction: ")

    def get_model_and_labels(self):
        if self.model_choice == "numbers":
            return model_numbers, labels_dict_numbers
        elif self.model_choice == "arabic":
            return model_arabic, labels_dict_arabic
        elif self.model_choice == "english_words":
            return model_english_words, labels_dict_english
        return model_alphabet, labels_dict_alphabet

    def add_space(self, event=None):
        self.sentence += " "
        if self.model_choice == "arabic":
            self.prediction_label.config(text="التوقع: مسافة")
        else:
            self.prediction_label.config(text="Prediction: ␣ (space)")
        self.sentence_label.config(text=self.sentence)
        self.esp.send(" ")

    def delete_char(self, event=None):
        self.sentence = self.sentence[:-1]
        if self.model_choice == "arabic":
            self.prediction_label.config(text="التوقع: حذف")
        else:
            self.prediction_label.config(text="Prediction: ⌫ (deleted)")
        self.sentence_label.config(text=self.sentence)
        self.esp.send("#")

    def reset_sentence(self):
        self.sentence = ""
        if self.model_choice == "arabic":
            self.prediction_label.config(text="التوقع: ")
            self.play_gtts("تمت إعادة تعيين الجملة")
        else:
            self.prediction_label.config(text="Prediction: ")
            engine.say("Sentence reset")
            engine.runAndWait()
        self.sentence_label.config(text=self.sentence)
        self.esp.send("*")

    def speak_sentence(self, event=None):
        if self.sentence.strip():
            if self.model_choice == "arabic":
                self.play_gtts(self.sentence)
            else:
                engine.say(self.sentence)
                engine.runAndWait()

    def play_gtts(self, text):
        try:
            tts = gTTS(text=text, lang='ar')
            filename = "temp_arabic_audio.mp3"
            tts.save(filename)
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
            os.remove(filename)
        except Exception as e:
            print(f"Error in Arabic TTS: {e}")

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        predicted_label = ""
        model, labels_dict = self.get_model_and_labels()

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Always draw landmarks regardless of camera state
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if self.is_camera_on:
                    features = []
                    
                    if self.model_choice == "english_words":
                        # 3D feature extraction for words model
                        landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                        wrist = landmarks[0]
                        landmarks -= wrist  # Center at wrist
                        mid_tip = landmarks[12]
                        scale = np.sqrt(mid_tip[0]**2 + mid_tip[1]**2 + mid_tip[2]**2)
                        if scale != 0:
                            landmarks /= scale  # Normalize
                        features = landmarks.flatten().tolist()
                    else:
                        # 2D feature extraction for other models
                        x_ = [lm.x for lm in hand_landmarks.landmark]
                        y_ = [lm.y for lm in hand_landmarks.landmark]
                        for lm in hand_landmarks.landmark:
                            features.extend([lm.x - min(x_), lm.y - min(y_)])

                    if len(features) == (63 if self.model_choice == "english_words" else 42):
                        prediction = model.predict([np.asarray(features)])
                        
                        if self.model_choice == "english_words":
                            predicted_label = prediction[0]
                        else:
                            predicted_label = labels_dict.get(int(prediction[0]), "")

        if self.is_camera_on:
            if predicted_label == self.last_prediction:
                self.repeat_count += 1
            else:
                self.repeat_count = 0
            self.last_prediction = predicted_label

            if self.repeat_count == 15:
                if self.model_choice == "english_words":
                    if len(predicted_label) > 0:
                        final_word = predicted_label + " " 
                        self.sentence += final_word
                        self.prediction_label.config(text=f"Prediction: {final_word}")
                        self.sentence_label.config(text=self.sentence)
                        self.esp.send(predicted_label)

                elif self.model_choice == "arabic":
                    final_char = arabic_map.get(predicted_label, predicted_label)
                    prediction_text = f"التوقع: {final_char}"
                    if predicted_label in ["space", "del"]:
                        getattr(self, "add_space" if predicted_label == "space" else "delete_char")()
                    elif predicted_label not in ["nothing", "del", "space"]:
                        self.sentence += final_char
                        self.prediction_label.config(text=prediction_text)
                        self.sentence_label.config(text=self.sentence)
                        self.esp.send(predicted_label)
                else:
                    final_char = predicted_label
                    prediction_text = f"Prediction: {final_char}"
                    if predicted_label in ["SPACE", "DEL","space","del"]:
                        getattr(self, "add_space" if predicted_label in ["SPACE","space"] else "delete_char")()
                    elif predicted_label not in ["NOTHING", "DEL", "SPACE", "Blank", "BLANK","space","del"]:
                        self.sentence += final_char
                        self.prediction_label.config(text=prediction_text)
                        self.sentence_label.config(text=self.sentence)
                        self.esp.send(predicted_label)

        # Update video feed
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((780, 580))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()