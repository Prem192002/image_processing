import cv2
import numpy as np

def image_processing_module(image_left, image_right):
 """Performs image processing on a pair of stereo images.

 Args:
 image_left: The left stereo image.
 image_right: The right stereo image.

 Returns:
 A tuple of the processed left and right images.
 """

 # Convert the images to grayscale.
 grayscale_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
 grayscale_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

 # Apply a Gaussian blur to the images.
 blurred_left = cv2.GaussianBlur(grayscale_left, (5, 5), 0)
 blurred_right = cv2.GaussianBlur(grayscale_right, (5, 5), 0)

 # Perform stereo matching on the images.
 disparity_map = cv2.stereoBM(blurred_left, blurred_right, cv2.STEREO_BM_NORMALIZED_CROSS)

 # Convert the disparity map to a depth map.
 depth_map = cv2.reprojectImageToDepth(disparity_map, None, cv2.CALIB_ZERO_DISPARITY, 1.0, 0.0)

 # Apply a threshold to the depth map to remove any invalid values.
 thresholded_depth_map = cv2.threshold(depth_map, 0.0, 1000.0, cv2.THRESH_BINARY)[1]

 # Apply a morphological operation to the thresholded depth map to fill in any holes.
 dilated_depth_map = cv2.dilate(thresholded_depth_map, np.ones((5, 5), np.uint8))

 # Erode the dilated depth map to remove any small objects.
 eroded_depth_map = cv2.erode(dilated_depth_map, np.ones((3, 3), np.uint8))

 # Convert the eroded depth map back to an image.
 depth_image = cv2.normalize(eroded_depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

 # Return the processed left and right images.
 return depth_image, image_left

if __name__ == "__main__":
 # Load the stereo images.
 image_left = cv2.imread("image_left.jpg")
 image_right = cv2.imread("image_right.jpg")

 # Process the stereo images.
 processed_left, processed_right = image_processing_module(image_left, image_right)

 # Display the original and processed images.
 cv2.imshow("Original Left Image", image_left)
 cv2.imshow("Processed Left Image", processed_left)
 cv2.imshow("Original Right Image", image_right)
 cv2.imshow("Processed Right Image", processed_right)

 # Wait for the user to press a key.
 cv2.waitKey(0)

 # Close all windows.
 cv2.destroyAllWindows()
import cv2
import numpy as np

# Load the object detection model
model = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the image
image = cv2.imread(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\input\uw1.jpeg')

# Convert the image to a blob
blob = cv2.dnn.blobFromImage(image, 1/255.0, (320, 320), (0, 0, 0), swapRB=True, crop=False)

# Set the input layer of the network
model.setInput(blob)

# Run the forward pass of the network
outputs = model.forward()

# Get the bounding boxes and confidence scores
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = detection[0:4]
        # Get the confidence score
        confidence = detection[5]
        # Get the class ID
        class_id = detection[6]

        # Only proceed if the confidence score is above a threshold
        if confidence > 0.5:
            # Add the bounding box and confidence score to the lists
            boxes.append([x1, y1, x2, y2])
            confidences.append(confidence)
            class_ids.append(class_id)

# Sort the bounding boxes by confidence score
indices = np.argsort(confidences)[::-1]

# Loop over the top-N bounding boxes
for i in indices:
    # Draw the bounding box on the image
    x1, y1, x2, y2 = boxes[i]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Label the bounding box with the class name
    text = str(class_names[class_ids[i]])
    cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
