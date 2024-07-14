import cv2
import numpy as np
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# Constants for conversion
Pixel_to_cm = 25  # Assumed calibration value: 25 pixels = 1 cm

# Function to calculate the boundary of the segmented object
def get_boundary(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to calculate the surface area of a circular object
def calculate_circular_area(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Assuming the largest contour is the object
        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        area = np.pi * (radius / Pixel_to_cm) ** 2
        return round(area, 2), (int(x - radius), int(y - radius), int(2 * radius), int(2 * radius))
    return 0, (0, 0, 0, 0)

# Function to overlay the segmented part on top of the original image
def overlay_segmented_part(background, mask, x, y):
    alpha_mask = mask.astype(float) / 255
    h, w = mask.shape
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha_mask * background[y:y+h, x:x+w, c] +
                                       (1 - alpha_mask) * background[y:y+h, x:x+w, c])
    return background

# Load the pretrained segmentation model
model = YOLO('runs/segment/train6/weights/best.pt')

# Run inference on an image
img = cv2.imread("/content/drive/MyDrive/gigazit/images.jpeg")
H, W, _ = img.shape
results = model(img)

for result in results:
    for mask in result.masks.data:
        mask = mask.numpy() * 255
        mask = cv2.resize(mask, (W, H))

        # Calculate circular area
        circ_area, circ_bbox = calculate_circular_area(mask)
        x, y, w, h = circ_bbox

        # Ensure mask and region of interest in the background are the same size
        resized_mask = cv2.resize(mask[y:y+h, x:x+w], (w, h))

        # Overlay the segmented part on top of the original image
        img = overlay_segmented_part(img, resized_mask, x, y)

        # Draw the enclosing circle
        cv2.circle(img, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 2)

        # Put the text with the area
        cv2.putText(img, f"A: {circ_area} cm^2", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save and show the image
cv2.imwrite('output_with_circular_areas.png', img)
cv2_imshow(img)
