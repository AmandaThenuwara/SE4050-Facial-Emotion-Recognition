# EfficientNetB0 Facial Emotion Recognition

## Overview

This component implements **Facial Emotion Recognition (FER)** using
**EfficientNetB0** with ImageNet pretrained weights. It classifies
facial expressions into seven FER-2013 classes: **Angry, Disgust, Fear,
Happy, Neutral, Sad, and Surprise**.

The implementation includes transfer learning, fine-tuning, final test
evaluation, uploaded-image inference, and **continuous live webcam
emotion recognition**.

## Model Architecture

``` text
Input Image (224 × 224 × 3)
        ↓
Preprocessing / Data Augmentation
        ↓
EfficientNetB0 (ImageNet Pretrained)
        ↓
GlobalAveragePooling2D
        ↓
Dropout
        ↓
Dense (7, Softmax)
        ↓
Emotion Prediction
```

## Dataset

The model uses the **FER-2013** dataset.

  Split               Images
  ----------------- --------
  Training folder     28,709
  Test folder          7,178

The original 48 × 48 grayscale images are loaded as RGB and resized to
**224 × 224 × 3** for EfficientNetB0. The original training folder is
split into training and validation subsets using an **80/20 split with
seed 42**. The test set is kept separate until final evaluation.

## Training Configuration

  Setting                     EfficientNetB0
  --------------------------- ---------------------------
  Optimizer                   Adam
  Initial Learning Rate       0.001
  Fine-tuning Learning Rate   1e-5
  Batch Size                  32
  Loss Function               Categorical Cross-Entropy
  Stage 1 Epochs              Max. 20
  Fine-tuning Epochs          Max. 15
  Pretrained Weights          ImageNet
  Fine-tuned Layers           Final 30 layers

Three experiments were evaluated: - **E1:** Frozen EfficientNetB0
baseline - **E2:** Fine-tuning of the final 30 layers - **E3:**
Class-weighted fine-tuning

**E2** was selected as the final configuration based on validation
performance.

## Final Test Results

  Metric                    Result
  ------------------- ------------
  Accuracy              **58.15%**
  Macro Precision       **56.66%**
  Macro Recall          **50.37%**
  Macro F1-score        **51.05%**
  Weighted F1-score     **56.99%**
  Macro ROC-AUC         **87.77%**

## Project Structure

``` text
SE4050-Facial-Emotion-Recognition/
├── app/
│   ├── app.py
│   └── realtime.py
├── models/
│   └── efficientnetb0/
│       └── IT23193772_EfficientNetB0_FER2013.keras
├── notebooks/
│   └── efficientnetb0/
│       └── IT23193772_EfficientNetB0_FER2013.ipynb
├── results/
│   └── efficientnetb0/
│       └── EfficientNetB0_Final_Test_Results.csv
├── src/
│   └── efficientnetb0/
│       ├── preprocessing.py
│       ├── predict.py
│       ├── test_model.py
│       ├── test_preprocessing.py
│       └── test_predictions.py
└── test_images/
```

> The trained `.keras` model may be excluded from Git depending on the
> repository's `.gitignore` configuration.

## Application Features

### Uploaded Image Prediction

The Streamlit application accepts JPG, JPEG, and PNG facial images and
displays: - Predicted emotion - Prediction confidence - Probabilities
for all seven classes - Probability chart - Detailed probability table

### Live Webcam Emotion Recognition

The application also supports continuous webcam-based inference:

``` text
Webcam
   ↓
Live Video Frames
   ↓
OpenCV Face Detection
   ↓
Face Cropping
   ↓
Image Preprocessing
   ↓
EfficientNetB0
   ↓
Emotion + Confidence
```

The face is detected and cropped before being passed to the
EfficientNetB0 prediction pipeline. Inference is performed periodically
rather than on every video frame to reduce processing overhead.

## Environment Setup

Activate the current Windows virtual environment:

``` powershell
C:\venvs\fer\Scripts\Activate.ps1
```

Install the prototype dependencies if required:

``` powershell
python -m pip install tensorflow streamlit pandas pillow streamlit-webrtc av opencv-python==4.11.0.86
```

The live face-detection implementation currently uses **OpenCV
4.11.0.86** because it provides the `CascadeClassifier` functionality
used by the webcam pipeline.

## Run the Application

From the project root:

``` powershell
C:\venvs\fer\Scripts\Activate.ps1
python -m streamlit run app/app.py
```

### Upload Image Mode

1.  Select **Upload Image**.
2.  Upload a JPG, JPEG, or PNG facial image.
3.  Click **Predict Emotion**.
4.  Review the emotion, confidence, and probabilities.

### Live Webcam Mode

1.  Select **Live Webcam**.
2.  Click **START**.
3.  Allow browser camera access.
4.  Position the face clearly in front of the camera.
5.  The system detects the face and continuously predicts the facial
    expression.
6.  Click **STOP** when finished.

## Preprocessing

For inference, images are: 1. Converted to RGB. 2. Resized to **224 ×
224**. 3. Converted to a NumPy array. 4. Expanded with a batch
dimension.

The resulting input shape is:

``` text
(1, 224, 224, 3)
```

The current pipeline does not manually divide pixel values by 255 before
inference, keeping preprocessing consistent with the EfficientNetB0
implementation used during model development.

## Testing

The component includes: - `test_model.py` --- verifies model loading. -
`test_preprocessing.py` --- verifies preprocessing and input
dimensions. - `test_predictions.py` --- tests predictions on multiple
images.

Formal performance metrics are obtained from the FER-2013 test set.

## Limitations

Predictions may be incorrect or uncertain for visually similar or
ambiguous expressions. FER-2013 is also class-imbalanced, particularly
for the Disgust class.

Real-world webcam images can differ from FER-2013 because of lighting,
camera quality, face orientation, background, facial scale, and subtle
or mixed expressions. Therefore, outputs should be interpreted as
**model estimates**, not definitive assessments of a person's emotional
state.

## Technologies

-   Python
-   TensorFlow / Keras
-   EfficientNetB0
-   NumPy
-   Pandas
-   Pillow
-   Streamlit
-   Streamlit WebRTC
-   OpenCV
-   FER-2013

