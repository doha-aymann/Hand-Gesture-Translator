import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier

save_folder = r'C:\Users\moham\Downloads\Phase1\Models'
os.makedirs(save_folder, exist_ok=True)

dataset_path = r'C:\Users\moham\Downloads\Phase1\DataSet\images_after'

data_file = os.path.join(save_folder, 'Arabic_Hand_Data.pkl')
model_file = os.path.join(save_folder, 'Arabic_Model.pkl')

def extract_hand_landmarks(img_path, hands):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            features = []

            for lm in hand_landmarks.landmark:
                features.append(lm.x - min(x_))
                features.append(lm.y - min(y_))

            if len(features) == 42:
                return features
    return None


def preprocess_dataset(dataset_path, output_file):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

    data = []
    labels = []
    label_dict = {}

    all_folders = sorted(os.listdir(dataset_path))
    for idx, folder in enumerate(all_folders):
        label_dict[idx] = folder  # Arabic label name

    for idx, folder in enumerate(all_folders):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in tqdm(os.listdir(folder_path), desc=f"Processing '{folder}'"):
            img_path = os.path.join(folder_path, file)
            features = extract_hand_landmarks(img_path, hands)
            if features:
                data.append(features)
                labels.append(idx)

    data = np.array(data)
    labels = np.array(labels)

    with open(output_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels, 'label_dict': label_dict}, f)

    print(f"\n✅ Preprocessing complete! Saved {len(data)} samples to '{output_file}'.")


def train_model(data_file, model_output):
    with open(data_file, 'rb') as f:
        dataset = pickle.load(f)

    data = dataset['data']
    labels = dataset['labels']
    label_dict = dataset['label_dict']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data, labels)

    with open(model_output, 'wb') as f:
        pickle.dump({'model': model, 'label_dict': label_dict}, f)

    print(f"\n✅ Training complete! Model saved to '{model_output}' with {len(label_dict)} classes.")


if __name__ == "__main__":
    preprocess_dataset(dataset_path, data_file)
    train_model(data_file, model_file)
