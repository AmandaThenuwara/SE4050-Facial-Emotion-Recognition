from PIL import Image
from preprocessing import preprocess_image


# Create a temporary test image
test_image = Image.new("RGB", (48, 48), color="white")

# Preprocess image
processed_image = preprocess_image(test_image)

print("Preprocessing successful!")
print("Original image size:", test_image.size)
print("Processed image shape:", processed_image.shape)
print("Data type:", processed_image.dtype)