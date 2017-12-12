## Advanced Lane Finding

<a href="https://imgflip.com/gif/20xnzf"><img src="https://i.imgflip.com/20xnzf.gif" title="made at imgflip.com"/></a><a href="https://imgflip.com/gif/20xo7l"><img src="https://i.imgflip.com/20xo7l.gif" title="made at imgflip.com"/></a>

## Overview

Software pipeline to identify lane lines and measure lane curvature from a video taken from a front-facing camera on a car. The high-level steps are as follows:
* Calibrate camera to remove distortion
* Apply thresholds on color and gradient features to isloate pixels of lane lines
* Apply perspective transform on region of interest 
* Classify left/right lane line pixels by searching image with sliding windows, or using the location of previously detected lane lines
* Generate lane line polynomial fit on the pixels detected from the previous step
* Warp the the polynomial fit back onto the original image 


### Calibrate Camera
Use OpenCV functions findChessboardCorners, drawChessboardCorners, calibrateCamera and chessboard images to compute calibration and distortion matrix to undistort each frame of video. Save the output to a pickle file for later use in the pipeline

![chessboard](processed_images/chessboard.png) 

## Pipeline

### 1. Distortion Correction
Use the OpenCV function undistort() and the pickled camera matrix and distortion coefficents to undistort the frame

![distortion](processed_images/undistorted.png)

### 2. Perspective Transform

Use trapezoidal region of interest and OpenCV function getPerspectiveTransform to get transform matrix and inverse matrix.

![perspective transform](processed_images/transform.png)

### 3. Apply Color and Gradient Thresholds

I manually tuned these color and gradient thresholds to be robust to lane color and varied lighting conditions. There are several combinations of thresholds that are detailed below:

![combined](processed_images/threshold1.png) ![combined](processed_images/threshold2.png)

### 4. Lane Line Detection and Polynomial Fit
To identify lane lines, use a sliding window technique to identify non-zero pixels (pixels that have passed the thresholding stage). After tuning, I discovered that 9 windows, 100 pixel window width, and 50 pixel minimum to shift window, gives good performance. Using the detected pixels, a lane line polynomial fit is calculated and saved for use in future frames.

![sliding window](processed_images/sliding_window.png)

A different approach is used to detect lane line pixels if there are 3 consecutive frames of "high confidence" lane line detections. A lane line is considered "detected with high confidence" when the 2 lines have similar curvature (<=400 meter curvature radius) and 500px < distance between lines < 750px. If 3 consecutive frames have a high confidence detection, then we average the 3 polynomial fits from those frames, and use the average as our region of interest (+/- 50 pixels) for detecting lane line pixels in this frame. 

![roi pixel detection](processed_images/roi_detection.png)

### 5. Final Image
Warp the fit from the rectified image back onto the original image. Add picture in picture video for visualization of raw detections

![final](processed_images/final.png)

## Dependencies

* Python 3
* OpenCV
* Numpy
* matplotlib
* Jupyter Notebook

Note: Udacity has a handy Anaconda environment that includes many of the dependencies used in the Self-Driving Car Nanodegree: [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md)

## Running the code 
The project is completed in a Jupyter notebook. 
To start Jupyter in your browser, run the following command at the terminal prompt and within your Python 3 environment:

`> jupyter notebook`
