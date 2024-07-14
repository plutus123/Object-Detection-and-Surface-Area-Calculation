from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the pretrained segmentation model
model = YOLO('runs/segment/train3/weights/best.pt')

# Constants for conversion
Pixel_to_cm = 25  # Assumed calibration value: 25 pixels = 1 cm

# Function to calculate the boundary of the segmented object
def get_boundary(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to overlay the segmented part on top of the original image
def overlay_segmented_part(background, mask, x, y):
    alpha_mask = mask.astype(float) / 255
    h, w = mask.shape
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha_mask * background[y:y+h, x:x+w, c] + 
                                       (1 - alpha_mask) * background[y:y+h, x:x+w, c])
    return background

# Run inference on an image
img = cv2.imread("test/162431.jpg")
H, W, _ = img.shape
results = model(img)

# List to store areas
areas = []

# Process each segmented object
for result in results:
    for mask in result.masks.data:
        mask = mask.cpu().numpy() * 255  # Move the tensor to the CPU and convert to NumPy array
        mask = cv2.resize(mask, (W, H))
        
        # Get boundaries of all segmented objects
        contours = get_boundary(mask)
        
        for contour in contours:
            # Get the bounding box of the object
            x, y, w, h = cv2.boundingRect(contour)
            
            # Ensure mask and region of interest in the background are the same size
            resized_mask = cv2.resize(mask[y:y+h, x:x+w], (w, h))
            
            # Overlay the segmented part on top of the original image
            img = overlay_segmented_part(img, resized_mask, x, y)
            
            # Draw the boundary
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
            
            # Calculate rectangular area
            area = (w / Pixel_to_cm) * (h / Pixel_to_cm)
            
            # Append area to list
            areas.append(area)
            
            # Display the area on the object with small font size
            text_size = cv2.getTextSize(f"A: {area:.2f} cm^2", cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            text_x = x + int((w - text_size[0]) / 2)
            text_y = y + int((h + text_size[1]) / 2)
            cv2.putText(img, f"A: {area:.2f} cm^2", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

# Save the image with overlay
cv2.imwrite('output_with_overlay.png', img)

# Display the image with areas displayed
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Print areas of all segmented objects
print("Areas of segmented objects (cm^2):")
for i, area in enumerate(areas):
    print(f"Object {i+1}: {area:.2f} cm^2")
