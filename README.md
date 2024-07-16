# Object-Detection-and-Surface-Area-Calculation

To improve the accuracy of object detection, a custom dataset was prepared using Roboflow for rectangular and circular objects. The YOLOv8 model was then trained on this custom dataset. The training command used is as follows:

```

!yolo task=segment mode=train model=yolov8m-seg.pt data=Rectangle_images/data.yaml epochs=100 imgsz=640 batch=8

```

# Challenge Description

The challenge involves:

1. Object Detection: Identifying and isolating the object present in an image.
2. Surface Area Calculation: Calculating the surface area of the detected object based on its shape assuming a flat object for simplicity.

# Approach

### Rectangular Object Detection and Surface Area Calculation

For images containing rectangular objects the following steps are performed:

1. Loading the Pretrained Segmentation Model: Using YOLOv8 for segmentation.
2. Inference on Image: Running the model to get segmented masks.
3. Boundary Extraction: Extracting contours of the segmented object.
4. Bounding Box Calculation: Determining the bounding box of the object.
5. Overlaying Segmented Part: Blending the segmented part onto the original image for visualization.
6. Area Calculation: Calculating the area using the formula: Area = Width * Height
7. Displaying Results: Annotating the image with the calculated area and visualizing the bounding box.

### Circular Object Detection and Surface Area Calculation

For images containing circular objects the following steps are performed:
1. Loading the Pretrained Segmentation Model: Using YOLOv8 for segmentation.
2. Inference on Image: Running the model to get segmented masks.
3. Boundary Extraction: Extracting contours of the segmented object.
4. Enclosing Circle Calculation: Determining the smallest enclosing circle of the object.
5. Overlaying Segmented Part: Blending the segmented part onto the original image for visualization.
6. Area Calculation: Calculating the area using the formula: Area = pi * Radius^2
7. Displaying Results: Annotating the image with the calculated area and visualizing the enclosing circle.

# Assumptions

• The object is flat and lying on a surface.
• The pixel-to-centimeter conversion factor is assumed to be 25 pixels = 1 cm.
• The largest contour detected is considered the object of interest.

# Output 
The script saves an annotated image (output_with_combined_boxes.png) with both rectangular and circular objects annotated with their respective bounding boxes or circles and calculates their surface areas in square centimeters. It also prints out the areas and types (rectangular or circular) of all segmented objects.

So for square objects I am getting an output like this:
![download](https://github.com/user-attachments/assets/c9ad574f-12d2-4775-a2bb-32531c32d1f3)


