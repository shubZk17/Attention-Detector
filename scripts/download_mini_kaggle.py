import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

api = KaggleApi()
api.authenticate()

# Folder to store raw data
raw_data_path = 'data/raw'
os.makedirs(raw_data_path, exist_ok=True)

# Download the full competition dataset (big zip file)
print("Downloading dataset ZIP...")
api.competition_download_files('state-farm-distracted-driver-detection', path=raw_data_path)

# Unzip it
zip_path = os.path.join(raw_data_path, 'state-farm-distracted-driver-detection.zip')

print("Unzipping...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(raw_data_path)

print("✅ Download and extraction complete.")
