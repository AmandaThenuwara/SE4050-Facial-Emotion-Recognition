# SE4050 Facial Emotion Recognition - Custom CNN

This branch contains the Custom Convolutional Neural Network (CNN) implementation for the SE4050 Deep Learning project.

The model is trained on the FER-2013 dataset to classify facial expressions into seven emotion categories.

## Supported Emotions

The model predicts:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Model Details

- Dataset: FER-2013
- Input size: `48 x 48`
- Color mode: Grayscale
- Input shape: `(48, 48, 1)`
- Number of classes: `7`
- Framework: TensorFlow / Keras
- Model file: `models/cnn/FER2013_Custom_CNN.keras`

The saved model already contains a `Rescaling(1/255)` layer, so input images should not be normalized manually before prediction.

## CNN Architecture

The Custom CNN contains three convolutional blocks:

- Conv2D + Batch Normalization + ReLU + Max Pooling
- Filters: `32`, `64`, `128`
- Global Average Pooling
- Dense layer with `128` units
- Dropout: `0.4`
- Softmax output layer with `7` classes

Total parameters: approximately `110k`.

## Project Structure

```text
models/
└── cnn/
    └── FER2013_Custom_CNN.keras

src/
└── cnn/
    ├── face_detection.py
    ├── predict.py
    └── preprocessing.py

test_images/
└── cnn/
    └── sample_face.png

notebooks/
└── cnn/
    ├── IT23175266_CNN_FER2013.ipynb
    └── artifacts/

app_cnn.py
test_cnn_model.py
requirements-cnn.txt
```

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements-cnn.txt
```

## Test the CNN Locally

Run:

```bash
python test_cnn_model.py
```

The test pipeline performs:

1. Face detection using OpenCV
2. Face cropping
3. Grayscale conversion
4. Resize to `48 x 48`
5. CNN inference
6. Emotion prediction
7. Confidence and class probability output

## Streamlit Application

Run the application with:

```bash
python -m streamlit run app_cnn.py
```

The application contains two modes.

### Upload Image

Upload a `.jpg`, `.jpeg`, or `.png` image.

The application will:

- Detect the face
- Crop the detected face
- Preprocess the image
- Predict the emotion
- Display the confidence score
- Display probabilities for all seven emotion classes

### Live Emotions

The Live Emotions mode performs real-time facial emotion recognition using the webcam.

The system continuously:

- Captures webcam frames
- Detects faces
- Converts detected faces to grayscale
- Resizes them to `48 x 48`
- Runs the CNN model
- Displays the predicted emotion and confidence on the video feed

Allow browser camera permission when prompted.

## Evaluation Results

The Custom CNN achieved the following results on the FER-2013 test set:

| Metric | Result |
|---|---:|
| Test Accuracy | 51.14% |
| Macro Precision | 41.92% |
| Macro Recall | 42.21% |
| Macro F1-score | 40.69% |
| Weighted F1-score | 48.84% |
| Macro ROC-AUC | 82.10% |
| Weighted ROC-AUC | 82.80% |

## Training Configuration

- Optimizer: Adam
- Initial learning rate: `0.001`
- Loss function: Categorical Crossentropy
- Batch size: `32`
- Maximum epochs: `30`
- Early stopping enabled
- ReduceLROnPlateau enabled
- Data augmentation:
  - Horizontal flip
  - Random rotation
  - Random zoom

## Notes

- The model was trained using TensorFlow `2.20.0`.
- The inference application can run on CPU.
- Webcam predictions may fluctuate because the model predicts independently from each frame.
- Face detection is handled using OpenCV Haar Cascade.