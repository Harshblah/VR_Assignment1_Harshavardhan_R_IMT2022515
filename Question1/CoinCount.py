import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image = cv2.imread(r"C:\Users\DELL\Desktop\VR\Assignment1\Question1\Input_Images\Img_3.jpg") # Load image

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Convert to HSV color space, to detect mettallic objects under different lighting conditions
                                             

lower_bound = np.array([0, 0, 30])   # Appropriate ranges to detect metallic objects
upper_bound = np.array([180, 150, 255])  
hsv_mask = cv2.inRange(hsv, lower_bound, upper_bound)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert filtered image to grayscale, removing colour info makes it easier for edge detection easier

blurred = cv2.GaussianBlur(gray, (15, 15), 5) # Apply Gaussian Blur to reduce noise

# Otsu's Thresholding
# Converts the grayscale image into a binary image (coins appear as white, background as black).
# Otsu’s method automatically selects the best threshold value.
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Canny Edge Detection 
edges = cv2.Canny(blurred, 100, 200)

combined_mask = cv2.bitwise_and(binary, hsv_mask)

# Morphological Closing removes small gaps, ensures contours are continous.
kernel = np.ones((5, 5), np.uint8)
processed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area and circularity
# Ensures not all circular objects but only coins of appropriate sizes are chosen
filtered_contours = []
for c in contours:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    if perimeter == 0:
        continue
    
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    
    if area > 500 and 0.6 < circularity < 1.3:  
        filtered_contours.append(c)

# Highlight detected coins
image_copy = image.copy()
cv2.drawContours(image_copy, filtered_contours, -1, (0, 255, 0), 2)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Segmentation Output
axes[0].imshow(processed, cmap='gray')
axes[0].set_title("Binary Segmentation Output")
axes[0].axis("off")

# Coin Count Output
axes[1].imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Coin Count Output - {len(filtered_contours)} Coins")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(r"C:\Users\DELL\Desktop\VR\Assignment1\Question1\Output_Images\Output_Img_3.jpg") 
plt.show()

print(f"Number of coins detected: {len(filtered_contours)}")
