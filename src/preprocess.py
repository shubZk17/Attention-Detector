import os
import cv2
import pickle
from tqdm import tqdm


RAW_DATA_PATH = "data/raw/imgs"
TRAIN_PATH = os.path.join(RAW_DATA_PATH, "train")
TEST_PATH = os.path.join(RAW_DATA_PATH, "test")
PROCESSED_DATA_PATH = "data/processed"
PROCESSED_FILE = os.path.join(PROCESSED_DATA_PATH, "mini_data.pkl")

IMG_SIZE = (24, 24)  

# Map folder names to binary labels: 0 = attentive (c0), 1 = distracted (c1-c9)
label_map = {
    "c0": 0, 
    "c1": 1, "c2": 1, "c3": 1, "c4": 1,
    "c5": 1, "c6": 1, "c7": 1, "c8": 1, "c9": 1
}

def preprocess_images():
    data = []
    labels = []

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"[ERROR] {TRAIN_PATH} not found.")

    print(f"[INFO] Loading data from: {TRAIN_PATH}")

    for label_folder in os.listdir(TRAIN_PATH):
        label_folder_path = os.path.join(TRAIN_PATH, label_folder)

        if not os.path.isdir(label_folder_path):
            continue

        if label_folder in label_map:
            label = label_map[label_folder]
        else:
            print(f"[WARNING] Unknown label for folder: {label_folder}, skipping...")
            continue

        for image_file in tqdm(os.listdir(label_folder_path), desc=f"Processing {label_folder}"):
            image_path = os.path.join(label_folder_path, image_file)

            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is not None:
                    image = cv2.resize(image, IMG_SIZE)
                    data.append(image)
                    labels.append(label)
                else:
                    print(f"[WARNING] Could not read image: {image_path}")
            except Exception as e:
                print(f"[ERROR] Exception processing {image_path}: {e}")

    # Save processed data
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    with open(PROCESSED_FILE, 'wb') as f:
        pickle.dump((data, labels), f)

    print(f"[INFO] Saved {len(data)} images to {PROCESSED_FILE}")

if __name__ == "__main__":
    preprocess_images()
