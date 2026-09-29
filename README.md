# SE4050 – Facial Emotion Recognition Using Deep Learning

This project was developed for the **SE4050 – Deep Learning** module.

The project implements and compares four deep-learning architectures for **Facial Emotion Recognition (FER)** using the **FER-2013 dataset**.

## Models

The following deep-learning models are implemented:

- Custom CNN
- VGG16
- ResNet50
- EfficientNetB0

The models classify facial expressions into seven emotion categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---
## Dataset

This project uses the **FER-2013 (Facial Expression Recognition 2013) dataset**.

### Dataset Link

**Kaggle – Challenges in Representation Learning: Facial Expression Recognition Challenge**

https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

### Dataset Characteristics

- Total images: 35,887
- Training images: 28,709
- Test images: 7,178
- Original image size: 48 × 48 pixels
- Image type: Grayscale
- Number of classes: 7
- Classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

The dataset should be downloaded separately from Kaggle and placed in the appropriate project directory before running the training notebooks.

## Project Structure

```text
SE4050-Facial-Emotion-Recognition/
│
├── notebooks/
│   ├── EDA / preprocessing notebooks
│   ├── Custom CNN notebook
│   ├── VGG16 notebook
│   ├── ResNet50 notebook
│   └── EfficientNetB0 notebook
│
├── src/
│   ├── preprocessing utilities
│   ├── prediction utilities
│   └── face detection utilities
│
├── models/
│   └── trained model files
│
├── results/
│   └── evaluation results
│
├── figures/
│   └── training graphs and confusion matrices
│
├── app.py
├── requirements.txt
└── README.md
```

> The exact files available may differ depending on the development branch.

---

# Setup Instructions

## 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/AmandaThenuwara/SE4050-Facial-Emotion-Recognition.git
cd SE4050-Facial-Emotion-Recognition
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Main technologies used include:

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- OpenCV
- Streamlit

---

# Model Training

The model-training experiments were primarily performed using **Google Colab**.

Open the required notebook from the `notebooks/` directory and execute the cells sequentially.

Each model follows the general workflow:

```text
Dataset Loading
      ↓
Data Preprocessing
      ↓
Data Augmentation
      ↓
Model Training
      ↓
Fine-Tuning (where applicable)
      ↓
Model Evaluation
      ↓
Save Results
```

The pretrained architectures use **ImageNet weights** for transfer learning.

---

# ResNet50 Implementation

The ResNet50 model uses the following pipeline:

```text
224 × 224 × 3 Input
        ↓
Data Augmentation
        ↓
ResNet50 Preprocessing
        ↓
ImageNet ResNet50
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.5)
        ↓
Dense (7, Softmax)
```

Training is performed in two stages:

**Stage 1:** Freeze the pretrained ResNet50 backbone and train the classification head.

**Stage 2:** Fine-tune selected final ResNet50 layers using a lower learning rate.

---

# Running the Streamlit Application

The trained ResNet50 model is also integrated into a Streamlit application for facial emotion prediction.

Activate the virtual environment and run:

```bash
streamlit run app.py
```

Then open the local URL displayed by Streamlit, normally:

```text
http://localhost:8501
```

The application allows the user to:

1. Capture a facial image using the camera.
2. Detect and crop the face using OpenCV.
3. Process the captured facial image.
4. Predict the emotion using the trained ResNet50 model.
5. Display the predicted emotion and confidence score.

---

# Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Final test accuracies obtained in this study:

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 51.14% |
| VGG16 | 61.70% |
| ResNet50 | 62.91% |
| EfficientNetB0 | 58.15% |

The experiments demonstrate differences in performance and training behaviour across the four architectures.

---


