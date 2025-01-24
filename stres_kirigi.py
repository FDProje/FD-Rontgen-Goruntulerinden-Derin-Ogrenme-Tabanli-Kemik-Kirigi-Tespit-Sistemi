import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
image_path = r"C:\Users\asyao\PycharmProjects\MICROFRACTURES\images\hairline56.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Enhance contrast and brightness to make the bones appear white
alpha = 3.0  # Increased contrast control
beta = 50    # Increased brightness control
enhanced_image = cv2.convertScaleAbs(original_image, alpha=alpha, beta=beta)

# Threshold the enhanced image to ensure bones are bright white
 , enhanced_image_thresh = cv2.threshold(enhanced_image, 180, 255, cv2.THRESH_BINARY)

# Apply Canny edge detection with lower thresholds for less detail
lower_threshold = 30  # Lower threshold for Canny (less detail)
upper_threshold = 100  # Upper threshold for Canny
canny_edges = cv2.Canny(original_image, lower_threshold, upper_threshold)

# Dilate the Canny edges to create thicker lines
kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(canny_edges, kernel, iterations=1)

# Convert the enhanced image to BGR to mark differences in red
marked_image = cv2.cvtColor(enhanced_image_thresh, cv2.COLOR_GRAY2BGR)

# Highlight the dilated Canny edges in red to detect the fracture line
marked_image[dilated_edges == 255] = [0, 0, 255]  # Red color for the fracture line

# Use matplotlib to display the images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Enhanced Image (White Bones)")
plt.imshow(enhanced_image_thresh, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Canny Edge Detection (Less Detail)")
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Marked Hairline Fracture (Red Line)")
plt.imshow(marked_image)
plt.axis('off')

plt.show()
