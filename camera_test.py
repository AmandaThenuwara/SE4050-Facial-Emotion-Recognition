import cv2


print("Searching for available cameras...\n")

for camera_id in range(5):

    print(f"Testing camera {camera_id}...")

    camera = cv2.VideoCapture(
        camera_id,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        print(f"Camera {camera_id}: Cannot open")
        camera.release()
        continue

    success, frame = camera.read()

    if success and frame is not None:
        print(
            f"Camera {camera_id}: WORKING "
            f"({frame.shape[1]}x{frame.shape[0]})"
        )
    else:
        print(
            f"Camera {camera_id}: Opened but cannot read frame"
        )

    camera.release()

print("\nCamera test completed.")