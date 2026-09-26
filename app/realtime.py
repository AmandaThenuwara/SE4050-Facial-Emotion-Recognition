import av
import cv2
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from predict import predict_emotion


class EmotionVideoProcessor(VideoProcessorBase):

    def __init__(self):

        # OpenCV built-in face detector
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Predict every few frames instead of every frame.
        # This reduces CPU/GPU workload.
        self.frame_count = 0
        self.prediction_interval = 5

        self.last_emotion = "Detecting..."
        self.last_confidence = 0.0


    def recv(self, frame):

        # Convert WebRTC frame to OpenCV image
        image = frame.to_ndarray(format="bgr24")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) > 0:

            # Use the largest detected face
            x, y, w, h = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            # Crop face
            face_image = image[
                y:y + h,
                x:x + w
            ]

            # Only run EfficientNetB0 every 5 frames
            if self.frame_count % self.prediction_interval == 0:

                try:

                    # BGR -> RGB
                    face_rgb = cv2.cvtColor(
                        face_image,
                        cv2.COLOR_BGR2RGB
                    )

                    # Convert to PIL
                    pil_image = Image.fromarray(face_rgb)

                    # Existing EfficientNetB0 function
                    emotion, confidence, probabilities = (
                        predict_emotion(pil_image)
                    )

                    self.last_emotion = emotion.upper()
                    self.last_confidence = confidence

                except Exception as error:

                    print(
                        f"Prediction error: {error}"
                    )

            # Draw face bounding box
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Prediction label
            label = (
                f"{self.last_emotion} "
                f"{self.last_confidence:.1f}%"
            )

            # Label background
            label_y = max(y - 35, 0)

            cv2.rectangle(
                image,
                (x, label_y),
                (x + w, y),
                (0, 255, 0),
                -1
            )

            # Draw emotion
            cv2.putText(
                image,
                label,
                (x + 5, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

        else:

            # No face found
            cv2.putText(
                image,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        self.frame_count += 1

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


def show_live_camera():

    webrtc_streamer(
        key="live-emotion-recognition",
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )