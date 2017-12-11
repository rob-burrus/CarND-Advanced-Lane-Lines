## Advanced Lane Finding

## Overview

Software pipeline to identify lane lines and measure lane curvature from a video taken from a front-facing camera on a car. The high-level steps are as follows:
* Calibrate camera to remove distortion
* Apply thresholds on color and gradient features to isloate pixels of lane lines
* Apply perspective transform on region of interest 
* Identify lane line pixels by searching image with sliding windows, or using the location of previously detected lane lines
* Generate lane line polynomial fit on the pixels detected from the previous step
* Warp the the polynomial fit back onto the original image 


### Calibrate Camera
Use OpenCV functions findChessboardCorners, drawChessboardCorners, calibrateCamera and chessboard images to compute calibration and distortion matrix to undistort each frame of video.
![chessboard](chessboard.png) 

## Pipeline

### 1. Distortion Correction
Use the OpenCV function undistort() and the pickled camera matrix and distortion coefficents to undistort the frame

![distortion](chessboard.png)

### 2. Apply Color and Gradient Thresholds

X and Y Gradients are used to detect steep edges that are likely to be lane lines

![gradients](grad_x.png) ![gradients](grad_y.png)

HLS Color space, specifically the S-channel provides a fairly robust way of detecting lane pixels, which often have high saturation

![color_space](color_space.png)

Combine thresholds to allow for robust lane line pixel detection in a variety of lighting conditions

![combined](combined.png)

### 3. Perspective Transform
Use trapezoidal region of interest and OpenCV function getPerspectiveTransform to get transform matrix and inverse matrix.

![perspective transform](transform.png)

### 4. Lane Line Detection and Polynomial Fit
To identify lane lines, use a sliding window technique to identify non-zero pixels (pixels that have passed the thresholding stage). After tuning, I discovered that 9 windows, 100 pixel window width, and 50 pixel minimum to shift window, gives good performance. Using the detected pixels, a lane line polynomial fit is calculated and saved for use in future frames.

![sliding window](sliding_window.png)

A different approach is used to detect lane line pixels if there are 3 consecutive frames of "high confidence" lane line detections. A lane line is considered "detected with high confidence" when the 2 lines have similar curvature (<=400 meter curvature radius) and 500px < distance between lines < 750px. If 3 consecutive frames have a high confidence detection, then we average the 3 polynomial fits from those frames, and use the average as our region of interest (+/- 50 pixels) for detecting lane line pixels in this frame. 

![roi pixel detection](roi_detection.png)

### 5. Final Image
Warp the fit from the rectified image back onto the original image. Add picture in picture video for visualization of raw detections

![final](final.png)


