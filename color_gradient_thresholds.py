
import cv2
import numpy as np


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output

def hls_select(img, channel=2, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    selected_channel = hls[:,:,channel]
    #selected_channel = cv2.equalizeHist(selected_channel)
    binary_output = np.zeros_like(selected_channel)
    binary_output[(selected_channel > thresh[0]) & (selected_channel <= thresh[1])] = 1
    return binary_output

def ycrcb_select(img, channel=2, thresh=(0, 255)):
    YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float)
    selected_channel = YCrCb[:,:,channel]
    #selected_channel = cv2.equalizeHist(selected_channel)
    binary_output = np.zeros_like(selected_channel)
    binary_output[(selected_channel > thresh[0]) & (selected_channel <= thresh[1])] = 1
    return binary_output

def rgb_select(img, channel=0, thresh=(0, 255)):
    rgb = img#cv2.cvtColor(img, cv2.COLOR_RGB2RGB).astype(np.float)
    selected_channel = rgb[:,:,channel]
    #selected_channel = cv2.equalizeHist(selected_channel)
    binary_output = np.zeros_like(selected_channel)
    binary_output[(selected_channel > thresh[0]) & (selected_channel <= thresh[1])] = 1
    return binary_output
