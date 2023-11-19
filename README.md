# Various Image Processing II

## Overview
This project involves image matching and homography estimation using Python with the OpenCV and scikit-image libraries. The key scripts are homography.py, run.py, and utils.py.

## Features
- Image Matching: The matchPics function in homography.py performs SIFT matching between two input images.
- RANSAC Homography Estimation: The computeH_ransac function in homography.py uses RANSAC to robustly estimate the homography matrix.
- Visualization: The visualize_box and visualize_match functions in utils.py provide visualizations of the bounding box and matching results, respectively.
- Composite Image Creation: The compositeH function in homography.py creates a composite image by warping a template image onto another image using the estimated homography.

## How to Use
1. Install Dependencies: Ensure you have the required libraries installed:
```
pip install numpy matplotlib opencv-python scikit-image
```
2. Clone the Repository: Clone the repository to your local machine.
```bash
git clone [repository_url]
cd [repository_directory]
```
3. Run the Script: Execute the run.py script to perform image matching, RANSAC homography estimation, and composite image creation.
```
python run.py
```

## Files
- homography.py: Contains functions for SIFT matching, homography computation, and composite image creation.
- run.py: Script to run the image matching and homography estimation process.
- utils.py: Utility functions for visualizing results.
- cv_desk.jpg: Sample image.
- cv_cover.jpg: Sample image.
- hp_cover.jpg: Sample image.

## Dependencies
- NumPy
- Matplotlib
- OpenCV
- scikit-image

## Author
Ismael Rekik
