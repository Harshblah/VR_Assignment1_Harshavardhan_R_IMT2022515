# VR_Assignment1_Harshavardhan_R_IMT2022515

# Task 1 - Coin Detection and Counting Using OpenCV  

## Overview  
This project detects and counts metallic coins in an image using computer vision techniques. The algorithm applies preprocessing, edge detection, segmentation, and contour filtering to identify coins accurately.  

## Workflow  

1. **Load Image:** The input image is read and processed.  
2. **Convert to HSV Color Space:** Helps in detecting metallic objects under varying lighting conditions.  
3. **Grayscale Conversion:** Removes color information, making edge detection more effective.  
4. **Noise Reduction:** Gaussian blur is applied to smooth the image.  
5. **Thresholding:** Otsu’s method converts the grayscale image into a binary format.  
6. **Edge Detection:** Canny edge detection is used to identify object boundaries.  
7. **Morphological Processing:** Closing operation ensures continuous contours.  
8. **Contour Detection & Filtering:** Contours are identified and filtered based on area and circularity to detect only coin-like objects.  
9. **Visualization:** Detected coins are outlined, and the count is displayed.

# Task 2 - Image Stitching and Processing

## Overview
This Python script stitches multiple images together into a panoramic image using OpenCV's stitching module. It also removes unnecessary black borders around the stitched image for a cleaner output.

## Features
- Loads images from a specified folder
- Detects keypoints using ORB (Oriented FAST and Rotated BRIEF)
- Stitches images together using OpenCV’s `Stitcher` class
- Removes black borders from the final stitched image
- Saves intermediate and final output images for visualization


## Dependencies  
Ensure the following libraries are installed before running the script:  
```bash
pip install numpy opencv-python matplotlib
```
## How to Run the Code
Follow these steps to run the code in your terminal:
Navigate to the directory containing the Python files:

 ```bash
cd /VR_Assignment1_Harshavardhan_R_IMT2022515
```
1. For task 1
 ```bash
python CoinCount.py
```

2. For task 2
 ```bash
python StitchImages.py
``` 
