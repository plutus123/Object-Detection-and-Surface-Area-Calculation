import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Constants for conversion
Pixel_to_cm = 25  # Assumed calibration value: 25 pixels = 1 cm

# Function to calculate the boundary of the segmented object
def get_boundary(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to calculate the surface area of a circular object
def calculate_circular_area(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for contour in contours:
        # Assuming the largest contour is the object
        (x, y), radius = cv2.minEnclosingCircle(contour)
        area = np.pi * (radius / Pixel_to_cm) ** 2
        areas.append((round(area, 2), (int(x - radius), int(y - radius), int(2 * radius), int(2 * radius))))
    return areas

# Function to overlay the segmented part on top of the original image
def overlay_segmented_part(background, mask, x, y):
    alpha_mask = mask.astype(float) / 255
    h, w = mask.shape

    # Calculate region of interest in the background
    roi = background[y:y+h, x:x+w]

    # Ensure alpha_mask and roi have the same dimensions
    alpha_mask_resized = cv2.resize(alpha_mask, (roi.shape[1], roi.shape[0]))

    # Perform overlay operation
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha_mask_resized * roi[:, :, c] +
                                        (1 - alpha_mask_resized) * background[y:y+h, x:x+w, c])

    return background

# Load the pretrained segmentation model
model = YOLO('runs/segment/train6/weights/best.pt')

# Run inference on an image
img = cv2.imread("/content/drive/MyDrive/gigazit/images.jpeg")
H, W, _ = img.shape
results = model(img)

# List to store areas
areas = []

# Process each segmented object
for result in results:
    for mask in result.masks.data:
        mask = mask.cpu().numpy() * 255
        mask = cv2.resize(mask, (W, H))

        # Calculate circular areas
        circular_areas = calculate_circular_area(mask)

        for circ_area, circ_bbox in circular_areas:
            x, y, w, h = circ_bbox

            # Overlay the segmented part on top of the original image
            img = overlay_segmented_part(img, mask[y:y+h, x:x+w], x, y)

            # Draw the enclosing circle
            cv2.circle(img, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 2)

            # Put the text with the area inside the circular object
            text_size = cv2.getTextSize(f"A: {circ_area} cm^2", cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            text_x = x + int((w - text_size[0]) / 2)
            text_y = y + int((h + text_size[1]) / 2)
            cv2.putText(img, f"A: {circ_area} cm^2", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # Append area to list
            areas.append(circ_area)

# Save the image with annotations
output_path = 'output_with_circular_areas.png'
cv2.imwrite(output_path, img)

# Display the image using matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

# Print areas of all segmented circular objects
print("Areas of segmented circular objects (cm^2):")
for i, area in enumerate(areas):
    print(f"Circular Object {i+1}: {area:.2f} cm^2")
