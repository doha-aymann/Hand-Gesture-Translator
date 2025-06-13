import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score
import mediapipe as mp

dataset_path = r'C:\Users\moham\Downloads\Phase1\DataSet\Train_Nums'

save_folder = r'C:\Users\moham\Downloads\Phase1\Models'

os.makedirs(save_folder, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

data = []  
labels = []  
label_dict = {}

all_folders = sorted(os.listdir(dataset_path))

for idx, folder in enumerate(all_folders):
    label_dict[idx] = folder  

for idx, folder in enumerate(all_folders):
    folder_path = os.path.join(dataset_path, folder)  
    label = folder

    for file in tqdm(os.listdir(folder_path), desc=f"Processing '{label}'"):
        img_path = os.path.join(folder_path, file)  
        img = cv2.imread(img_path)  

        if img is None:  
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:  
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                features = []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    features.append(lm.x - min(x_))  
                    features.append(lm.y - min(y_)) 

                if len(features) == 42:  
                    data.append(features)
                    labels.append(idx)  
        else:
            features = [0] * 42  
            data.append(features)
            labels.append(10) 

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

model_save_path = os.path.join(save_folder, 'digits_model.p')
with open(model_save_path, 'wb') as f:
    pickle.dump({'model': model, 'labels_dict': label_dict}, f)

print(f"Training complete! Model saved at {model_save_path}.")
