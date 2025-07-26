🧠 Attention Detection using Webcam

This project detects user distraction using webcam footage and machine learning. It classifies whether the person is attentive or distracted based on eye behavior captured through images.




## 🚀 Features

- Detects user attention using image classification
- Preprocessing includes flattening and label encoding
- Trained on a dataset of 22,000 images
- Uses SVM model (can be extended to Neural Networks)
- Real-time inference with webcam (OpenCV)

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/attention-detection.git
   cd attention-detection

2. **Set up a virtual environment:**
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On Linux/macOS

3. **Install dependencies:** 

   pip install -r requirements.txt

4. **Preprocess the data:**
   Ensure your image data is inside data/raw/imgs/train/ and labeled by folders (e.g., c0, c1, ..., c9).

   code:
   python src/preprocess.py

5. **Train the model:**

   code:
   python src/train.py

6. **Run real-time attention detection:**

   code:
   python src/infer.py