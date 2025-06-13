import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import mediapipe as mp

# Path for the data
dataset_path = 'C:/Users/hp/Downloads/archive/asl_alphabet_train/asl_alphabet_train'

# Mediapipe setup for the pretrained hand detection with suitable parameters
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Prepare storage
# for convert flatten vector 42 feature
data = []
# save indecis for later use in model train
labels = []
# maps label index to actual name (e.g., 0: 'A', 26: 'space')
label_dict = {}  

# get all subfolder names sorted correctly
all_folders = sorted(os.listdir(dataset_path))

# now mapping each letter to index
# example: 0 -> 'A', 1 -> 'B'
for idx, folder in enumerate(all_folders):
    label_dict[idx] = folder.upper()


for idx, folder in enumerate(all_folders):
    
    # access the folder images for each letter
    folder_path = os.path.join(dataset_path, folder)
    # each letter
    label = folder.upper()

    # loop in each image in each letter file
    for file in tqdm(os.listdir(folder_path), desc=f"Processing '{label}'"):
        
        # now get the path of each image
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)

        # if cv2 faild to read the current image
        if img is None:
            continue

        # cv2 read each image in BGR format but mediapipe expect RGB so will convert it 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # send the image to mediapipe to detect the (21) landmarks using pretrained model inside it 
        results = hands.process(img_rgb)

        # start collect landmarks if the hand is detected
        if results.multi_hand_landmarks:

            # ((results.multi_hand_landmarks)) -> this is for if multi hands used
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                # for all x,y coordinates
                features = []

                # loop in each landmarks for each hand
                for lm in hand_landmarks.landmark:
                    # detect each x,y positions
                    x_.append(lm.x)
                    y_.append(lm.y)

                # loop again to normalize the x,y coordinates to make the hand independent of hand position in any image
                for lm in hand_landmarks.landmark:
                    features.append(lm.x - min(x_))
                    features.append(lm.y - min(y_))

                # now check if x,y as 21,21 converted to one vector 42 value
                if len(features) == 42:
                    # if ok save the data to train the model later and the labels for model target column
                    data.append(features)
                    labels.append(idx)

# Convert to numpy arrays to trainn the model
data = np.array(data)
labels = np.array(labels)


# used ml models not nn for -> data is cleaned , low dimensional, 
# structured in tabular 42 number each , pre extracted features by landmarks 

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(data, labels)

# Save the model and label dictionary
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'labels_dict': label_dict}, f)

print("Training complete! Model saved with 29 classes.")
