# VGG16 Facial Emotion Recognition

## ImageNet Transfer Learning + Fine-Tuning

This component implements **Facial Emotion Recognition (FER)** using a **VGG16 convolutional neural network with ImageNet-pretrained weights, transfer learning, and fine-tuning**.

The model was trained and evaluated using the **FER-2013 dataset** and classifies facial expressions into seven categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

In addition to model development and evaluation, the component includes an interactive **Streamlit application** supporting:

- Single-image emotion prediction
- Multiple-image emotion prediction
- Automatic face detection
- Automatic face cropping
- Multiple-face processing
- Live webcam emotion recognition

---

# 1. Model Overview

The VGG16 implementation uses **transfer learning** instead of training the complete convolutional network from scratch.

An **ImageNet-pretrained VGG16 backbone** is used for feature extraction. The original VGG16 classification layers are excluded using `include_top=False`, and a custom classification head is added for the seven FER-2013 facial-expression classes.

Training is performed in two stages:

1. **Stage 1 – Transfer Learning:** The pretrained VGG16 backbone is frozen while the new classification head is trained.
2. **Stage 2 – Fine-Tuning:** The final VGG16 layers are unfrozen and trained using a smaller learning rate.

The final trained model is then used for image and webcam-based facial-expression classification.

---

# 2. Dataset

The model uses the **FER-2013 facial expression dataset**.

| Property | Value |
|---|---:|
| Total Images | 35,887 |
| Training Images | 28,709 |
| Test Images | 7,178 |
| Original Resolution | 48 × 48 |
| Original Image Type | Grayscale |
| Model Input Resolution | 224 × 224 × 3 |
| Number of Classes | 7 |

The seven emotion classes are:

```text
angry
disgust
fear
happy
neutral
sad
surprise
```

The original training directory is divided into training and validation subsets using:

```text
Validation Split = 20%
Random Seed = 42
```

The original FER-2013 test directory remains separate and is used for final model evaluation.

---

# 3. VGG16 Architecture

The model uses an ImageNet-pretrained VGG16 backbone with a custom classification head.

## Architecture Pipeline

```text
Input Image
224 × 224 × 3
      │
      ▼
Data Augmentation
      │
      ▼
VGG16 preprocess_input
      │
      ▼
ImageNet-Pretrained VGG16
include_top = False
      │
      ▼
GlobalAveragePooling2D
      │
      ▼
Dense
256 Units + ReLU
      │
      ▼
Dropout
0.5
      │
      ▼
Dense
7 Units + Softmax
      │
      ▼
Facial Expression Classification
```

The final Softmax layer produces probabilities for:

```text
Angry | Disgust | Fear | Happy | Neutral | Sad | Surprise
```

---

# 4. Data Preprocessing

The FER-2013 images are resized from their original resolution to:

```text
224 × 224 × 3
```

Images are loaded in RGB format because the ImageNet-pretrained VGG16 model expects three-channel image input.

The main preprocessing configuration is:

```text
Image Size       : 224 × 224
Batch Size       : 32
Validation Split : 20%
Random Seed      : 42
Label Mode       : Categorical
Color Mode       : RGB
```

The test dataset is loaded using:

```text
shuffle = False
```

to preserve the relationship between test images and their labels during evaluation.

---

# 5. Data Augmentation

Training-time data augmentation is used to improve model generalization.

The augmentation pipeline includes:

```text
Random Horizontal Flip
Random Rotation     = 0.05
Random Zoom         = 0.10
Random Translation  = 0.05
```

Data augmentation is applied during training only.

The validation and test datasets are not randomly augmented.

---

# 6. Transfer Learning

## Stage 1 – Classification Head Training

The VGG16 backbone is initialized using:

```python
VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
```

During Stage 1, the VGG16 backbone is frozen.

Only the custom classification head is trained.

### Stage 1 Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Initial Learning Rate | 0.001 (1e-3) |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |
| Maximum Epochs | 20 |
| Early Stopping | Yes |
| ReduceLROnPlateau | Yes |
| Pretrained Weights | ImageNet |

During this stage:

```text
Total Parameters     : 14,847,815
Trainable Parameters :    133,127
```

---

# 7. Fine-Tuning

After training the classification head, part of the VGG16 backbone is unfrozen.

The final four VGG16 layers are made trainable:

```text
block5_conv1
block5_conv2
block5_conv3
block5_pool
```

The model is then recompiled using a lower learning rate.

### Fine-Tuning Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Fine-Tuning Learning Rate | 1e-5 |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |
| Maximum Epochs | 10 |
| Early Stopping | Yes |
| ReduceLROnPlateau | Yes |

The fine-tuned model contains:

```text
Trainable Parameters : 7,212,551
```

The lower learning rate allows the pretrained features to be adjusted gradually for facial-expression recognition.

---

# 8. Training Callbacks

Two primary callbacks are used during training.

## EarlyStopping

```text
Monitor              : val_loss
Patience             : 5
Restore Best Weights : True
```

Early stopping prevents unnecessary training after validation performance stops improving.

## ReduceLROnPlateau

```text
Monitor               : val_loss
Factor                : 0.2
Patience              : 3
Minimum Learning Rate : 1e-7
```

The learning rate is reduced when validation loss stops improving.

---

# 9. Final VGG16 Test Results

The final fine-tuned VGG16 model was evaluated on the **unseen FER-2013 test set containing 7,178 images**.

| Metric | Final Result |
|---|---:|
| Test Accuracy | **61.70%** |
| Test Loss | **1.0240** |
| Macro Precision | **63.46%** |
| Macro Recall | **51.92%** |
| Macro F1-score | **52.13%** |
| Weighted Precision | **62.35%** |
| Weighted Recall | **61.70%** |
| Weighted F1-score | **60.87%** |

---

# 10. Per-Class Performance

| Emotion | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Angry | 56.02% | 48.54% | 52.01% | 958 |
| Disgust | 80.00% | 3.60% | 6.90% | 111 |
| Fear | 47.80% | 29.69% | 36.63% | 1,024 |
| Happy | 83.66% | 85.12% | 84.38% | 1,774 |
| Neutral | 55.41% | 67.32% | 60.78% | 1,233 |
| Sad | 44.36% | 58.30% | 50.38% | 1,247 |
| Surprise | 76.99% | 70.88% | 73.81% | 831 |

The model performs particularly well for **Happy** and **Surprise**.

The most difficult classes are **Fear** and particularly **Disgust**.

The difference between the macro F1-score (**52.13%**) and weighted F1-score (**60.87%**) indicates that model performance is not uniform across all seven emotion classes.

---

# 11. Confusion Matrix

The VGG16 confusion matrix is available at:

```text
figures/vgg16/confusion_matrix.png
```

The normalized confusion matrix is available at:

```text
figures/vgg16/normalized_confusion_matrix.png
```

These matrices provide a class-level view of correct predictions and misclassifications.

---

# 12. Training Curves

Training and validation accuracy/loss curves are stored under:

```text
figures/vgg16/
```

including:

```text
stage1_accuracy.png
stage1_loss.png
finetune_accuracy.png
finetune_loss.png
```

These figures are used to analyze convergence, generalization and potential overfitting during both training stages.

---

# 13. Reproducibility Verification

The final VGG16 model was saved and independently reloaded for evaluation.

The reproduced model achieved the same test accuracy:

```text
Original Accuracy : 0.617024
Reloaded Accuracy : 0.617024
```

The other reproduced metrics were also consistent with the original final evaluation.

This confirms that the saved model can be loaded and evaluated without retraining.

The reproducibility script is:

```text
src/vgg16/verify_reproducibility.py
```

---

# 14. Saved Model

The final trained model is stored at:

```text
models/
└── vgg16/
    └── IT23220560_VGG16_FER2013.keras
```

The saved model contains the VGG16 preprocessing operation.

Therefore, application inference should **not apply `preprocess_input` a second time**.

---

# 15. Single-Image Prediction

Single-image prediction is implemented using:

```text
src/vgg16/predict.py
```

The prediction pipeline is:

```text
Input Image
      ↓
RGB Conversion
      ↓
Resize to 224 × 224
      ↓
Saved VGG16 Model
      ↓
Softmax
      ↓
Predicted Expression
```

Example terminal command:

```powershell
python src/vgg16/test_prediction.py "path/to/image.jpg"
```

Example output:

```text
Predicted Emotion : HAPPY
Confidence        : 94.27%
```

---

# 16. Automatic Face Detection and Cropping

Real-world images can contain significant background information and faces that occupy only a small portion of the photograph.

To make real-world inputs more similar to the face-focused FER-2013 images, automatic face detection and cropping are included in the prototype.

The implementation uses an **OpenCV Haar Cascade face detector**.

The pipeline is:

```text
Real-World Image
       ↓
OpenCV Face Detection
       ↓
Face Bounding Box
       ↓
15% Padding
       ↓
Face Crop
       ↓
Resize to 224 × 224
       ↓
VGG16
       ↓
Expression Prediction
```

The implementation is available in:

```text
src/vgg16/face_detection.py
```

The face detector was tested using:

```text
opencv-python==4.11.0.86
```

---

# 17. Multiple-Face Recognition

The prototype supports multiple faces within a single photograph.

```text
                   Input Image
                        ↓
                  Face Detection
                        ↓
             ┌──────────┼──────────┐
             ↓          ↓          ↓
           Face 1     Face 2     Face 3
             ↓          ↓          ↓
            Crop       Crop       Crop
             └──────────┼──────────┘
                        ↓
                  VGG16 Prediction
                        ↓
             Individual Predictions
```

Each detected face receives its own predicted facial-expression class and probability.

---

# 18. Streamlit Application

An interactive Streamlit prototype is provided in:

```text
src/vgg16/app.py
```

The application supports three main prediction modes:

```text
1. Single Image
2. Multiple Images
3. Live Webcam
```

All three modes reuse the trained VGG16 model.

---

# 19. Single Image Mode

The Single Image mode allows the user to upload one photograph.

Supported formats include:

```text
.jpg
.jpeg
.png
```

The application performs:

```text
Uploaded Image
      ↓
Face Detection
      ↓
Face Cropping
      ↓
VGG16 Prediction
      ↓
Predicted Expression
      ↓
Probability Distribution
```

If multiple faces are present, each detected face is processed separately.

---

# 20. Multiple Images Mode

The Multiple Images mode allows several images to be uploaded simultaneously.

```text
Multiple Images
      ↓
Face Detection
      ↓
Face Cropping
      ↓
VGG16 Prediction
      ↓
Individual Results
      ↓
Overall Prediction Summary
```

Each detected face is classified independently.

---

# 21. Live Webcam Emotion Recognition

The Streamlit application supports continuous live webcam prediction using **WebRTC**.

The live processing pipeline is:

```text
Live Webcam
     ↓
Video Frames
     ↓
OpenCV Face Detection
     ↓
Face Crop
     ↓
Resize to 224 × 224
     ↓
VGG16
     ↓
Expression + Probability
     ↓
Annotated Live Video
```

A bounding box is displayed around the detected facial region together with the predicted expression and probability.

Because VGG16 inference is computationally expensive when running on CPU, predictions can be performed periodically rather than on every video frame.

---

# 22. Standalone Webcam Application

A standalone OpenCV webcam implementation is also included:

```text
src/vgg16/webcam_prediction.py
```

This allows real-time emotion recognition without launching Streamlit.

The webcam pipeline is:

```text
Webcam
   ↓
Face Detection
   ↓
Face Crop
   ↓
VGG16
   ↓
Expression Prediction
   ↓
Annotated Webcam Frame
```

---

# 23. VGG16 Component Structure

```text
SE4050-Facial-Emotion-Recognition/
│
├── models/
│   └── vgg16/
│       └── IT23220560_VGG16_FER2013.keras
│
├── notebooks/
│   └── vgg16/
│       └── IT23220560__VGG16_FER2013.ipynb
│
├── src/
│   └── vgg16/
│       ├── README.md
│       ├── preprocessing.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       ├── verify_reproducibility.py
│       ├── predict.py
│       ├── face_detection.py
│       ├── webcam_prediction.py
│       ├── app.py
│       ├── test_preprocessing.py
│       ├── test_architecture.py
│       ├── test_training_config.py
│       ├── test_prediction.py
│       └── test_face_detection.py
│
├── results/
│   └── vgg16/
│       ├── VGG16_Final_Test_Results.csv
│       └── VGG16_Local_Evaluation_Results.csv
│
├── figures/
│   └── vgg16/
│       ├── stage1_accuracy.png
│       ├── stage1_loss.png
│       ├── finetune_accuracy.png
│       ├── finetune_loss.png
│       ├── confusion_matrix.png
│       └── normalized_confusion_matrix.png
│
├── requirements.txt
└── README.md
```

---

# 24. Requirements

The VGG16 component uses:

```text
Python
TensorFlow
Keras
NumPy
Pandas
Scikit-learn
Matplotlib
Pillow
OpenCV
Streamlit
streamlit-webrtc
PyAV
```

The OpenCV version used for the face-detection implementation is:

```text
opencv-python==4.11.0.86
```

The required Python dependencies should be listed in the project's:

```text
requirements.txt
```

---

# 25. How to Run the Application

## Step 1 — Clone the Repository

Clone the project:

```powershell
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Navigate into the repository:

```powershell
cd SE4050-Facial-Emotion-Recognition
```

---

## Step 2 — Create a Virtual Environment

Create a Python virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, PowerShell should show:

```text
(.venv) PS C:\...\SE4050-Facial-Emotion-Recognition>
```

---

## Step 3 — Install Dependencies

Install the required dependencies:

```powershell
pip install -r requirements.txt
```

If the Streamlit live webcam dependencies are not already included, install:

```powershell
pip install streamlit-webrtc av
```

The tested OpenCV version can be installed using:

```powershell
pip install opencv-python==4.11.0.86
```

---

## Step 4 — Check the Saved Model

Make sure the trained VGG16 model exists at:

```text
models/vgg16/IT23220560_VGG16_FER2013.keras
```

The application uses this saved model directly.

**Retraining is not required to run the application.**

---

## Step 5 — Start the Streamlit Application

From the root project directory, run:

```powershell
streamlit run src/vgg16/app.py
```

Streamlit should start the application and display a local URL similar to:

```text
Local URL: http://localhost:8501
```

The application will normally open automatically in the default browser.

If it does not open automatically, open the displayed local URL manually.

---

# 26. Using the Application

The sidebar provides:

```text
Prediction Settings

○ Single Image
○ Multiple Images
○ Live Webcam
```

---

## Single Image

Select:

```text
Single Image
```

Upload a `.jpg`, `.jpeg`, or `.png` image.

The application automatically performs:

```text
Upload
   ↓
Face Detection
   ↓
Face Crop
   ↓
VGG16
   ↓
Prediction
```

The interface displays:

- Original image
- Detected facial region
- Cropped face
- Predicted expression
- Model probability
- Seven-class probability distribution

---

## Multiple Images

Select:

```text
Multiple Images
```

Upload several images.

Each image is processed independently.

The application displays:

- Detected faces
- Individual predictions
- Model probabilities
- Overall prediction summary
- Predicted-expression distribution

---

## Live Webcam

Select:

```text
Live Webcam
```

Click the webcam **START** control if required.

When the browser requests permission to access the camera, select:

```text
Allow
```

The application will continuously process webcam frames.

The live pipeline is:

```text
Camera
   ↓
Frame
   ↓
Face Detection
   ↓
Face Crop
   ↓
VGG16
   ↓
Expression + Probability
   ↓
Live Display
```

For better webcam results:

- Keep the face clearly visible.
- Look toward the camera.
- Use sufficient lighting.
- Avoid covering the face.
- Keep the face reasonably close to the camera.

---

# 27. Run Standalone Live Webcam Prediction

The standalone webcam implementation can be run using:

```powershell
python src/vgg16/webcam_prediction.py
```

A separate OpenCV webcam window will open.

The system will continuously:

```text
Detect Face
    ↓
Crop Face
    ↓
Predict Expression
    ↓
Display Bounding Box
    ↓
Display Prediction
```

Press:

```text
Q
```

to close the webcam window.

---

# 28. Run Single-Image Prediction from Terminal

Single-image inference can also be tested without Streamlit.

Run:

```powershell
python src/vgg16/test_prediction.py "path/to/image.jpg"
```

Example:

```powershell
python src/vgg16/test_prediction.py "img/garfield_test.jpg"
```

Example output:

```text
============================================================
FACIAL EMOTION PREDICTION
============================================================

Predicted Emotion : HAPPY
Confidence        : 94.27%
```

---

# 29. Test Face Detection

Face detection can be tested separately using:

```powershell
python src/vgg16/test_face_detection.py "path/to/image.jpg"
```

Example:

```powershell
python src/vgg16/test_face_detection.py "img/garfield_test.jpg"
```

Generated face-detection outputs are stored under:

```text
outputs/
└── face_detection_test/
```

For example:

```text
garfield_test_detected.jpg
garfield_test_face_1.jpg
```

---

# 30. Evaluate the Saved VGG16 Model

To evaluate the saved model:

```powershell
python src/vgg16/evaluate.py
```

The FER-2013 test dataset must be available locally for this step.

The evaluation produces metrics including:

```text
Test Loss
Accuracy
Macro Precision
Macro Recall
Macro F1-score
Weighted Precision
Weighted Recall
Weighted F1-score
Classification Report
Confusion Matrix
```

---

# 31. Verify Reproducibility

Run:

```powershell
python src/vgg16/verify_reproducibility.py
```

The expected result is:

```text
REPRODUCIBILITY CHECK: PASSED
```

This verifies that the saved model produces results consistent with the original evaluation.

---

# 32. Stop the Streamlit Application

To stop Streamlit, return to the PowerShell terminal where the application is running and press:

```text
Ctrl + C
```

---

# 33. Quick Start

If Python is already installed, the main application can be started using:

```powershell
git clone <YOUR-GITHUB-REPOSITORY-URL>

cd SE4050-Facial-Emotion-Recognition

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

streamlit run src/vgg16/app.py
```

Then select:

```text
Single Image
Multiple Images
Live Webcam
```

from the sidebar.

---

# 34. Generalization and Limitations

The final VGG16 model achieved:

```text
Test Accuracy = 61.70%
```

on the unseen FER-2013 test set.

However, performance on arbitrary real-world photographs may differ from FER-2013 test performance.

FER-2013 contains low-resolution, face-focused images, while real-world images may contain:

- Complex backgrounds
- Different lighting conditions
- Different camera quality
- Different image resolution
- Non-frontal faces
- Occlusion
- Glasses
- Multiple people
- Small facial regions

Automatic face detection and cropping are therefore included in the prototype to make application inputs more face-focused.

However, cropping does not guarantee a correct prediction because the trained classifier still has generalization limitations.

---

# 35. Interpretation of Predictions

The application performs **facial-expression classification**.

For example:

```text
HAPPY
94.27%
```

means:

> The model assigned 94.27% of its output probability to the `Happy` facial-expression class.

It does **not** mean:

> The person is 94.27% happy.

The model classifies visible facial patterns according to the seven FER-2013 categories and does not determine a person's actual internal emotional state.

---

# 36. Technologies

## Deep Learning

```text
TensorFlow
Keras
VGG16
ImageNet Transfer Learning
```

## Data Processing

```text
NumPy
Pandas
Scikit-learn
```

## Computer Vision

```text
OpenCV
Pillow
```

## Application

```text
Streamlit
streamlit-webrtc
PyAV
```

## Development

```text
Python
Google Colab
Visual Studio Code
Git
GitHub
```

---

# 37. Component Status

| Component | Status |
|---|:---:|
| FER-2013 preprocessing | ✅ |
| Data augmentation | ✅ |
| VGG16 architecture | ✅ |
| ImageNet transfer learning | ✅ |
| Fine-tuning | ✅ |
| Test evaluation | ✅ |
| Classification report | ✅ |
| Confusion matrix | ✅ |
| Training curves | ✅ |
| Saved model | ✅ |
| Reproducibility verification | ✅ |
| Single-image prediction | ✅ |
| Multiple-image prediction | ✅ |
| Automatic face detection | ✅ |
| Automatic face cropping | ✅ |
| Multiple-face processing | ✅ |
| Streamlit UI | ✅ |
| Standalone webcam prediction | ✅ |
| Live Streamlit webcam prediction | ✅ |

---

# 38. Summary

The complete VGG16 workflow is:

```text
                 FER-2013
                     ↓
                 Preprocessing
                     ↓
               Data Augmentation
                     ↓
             ImageNet VGG16
                     ↓
              Transfer Learning
                     ↓
                Fine-Tuning
                     ↓
              Final Evaluation
                     ↓
              61.70% Accuracy
                     ↓
                Saved Model
                     ↓
             Application Layer
                     ↓
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
 Single Image   Multiple Images  Live Webcam
       │             │             │
       └─────────────┼─────────────┘
                     ↓
               Face Detection
                     ↓
                Face Cropping
                     ↓
                    VGG16
                     ↓
           Expression Prediction
```

This implementation demonstrates the complete deep-learning workflow from model development and evaluation to practical real-time application integration.

---

