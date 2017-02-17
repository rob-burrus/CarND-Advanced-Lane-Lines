#image generator
import cv2
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


dist_pickle = pickle.load(open("camera_cal/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
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

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
     # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def process_image(img, line_tracker):
    #write_name = './video_images/test' + str(line_tracker.index) + '.jpg'
    #line_tracker.index = line_tracker.index + 1
    #cv2.imwrite(write_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #
    #Undistort image
    #
    img = cv2.undistort(img, mtx, dist,None, mtx)

    #
    # Apply Gradient and Color Thresholds
    #
    ksize = 3
    gradx_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady_binary = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(25, 100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
    color_binary = hls_select(img, thresh=(170, 255))
    combined_binary = np.zeros_like(gradx_binary)
    combined_binary[(color_binary == 1) | ((gradx_binary == 1) & (grady_binary == 1)) ] = 1
    #plt.imshow(combined_binary, cmap='gray')
    #plt.show()
    #f, imgs = plt.subplots(2, 2, sharex=True, sharey=True)
    #f.tight_layout()
    #imgs[0,0].imshow(gradx_binary, cmap='gray')
    #imgs[0,0].set_title('grad_x')
    #imgs[0,1].imshow(color_binary, cmap='gray')
    #imgs[0,1].set_title('color')
    #imgs[1,0].imshow(combined_binary, cmap='gray')
    #imgs[1,0].set_title('cobined')

    #
    #Perspective Transform
    #
    img_size = (img.shape[1], img.shape[0])
    bot_width = .76
    mid_width = .08
    height_pct = .62
    bottom_trim = .935
    src = np.float32([[img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(.5-bot_width/2), img.shape[0]*bottom_trim] ])
    offset = img_size[0]*.2
    dst = np.float32([[offset,0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_binary = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)

    #
    #Find lane lines
    #
    result, out_img = line_tracker.find_lane_lines(img, warped_binary, Minv)

    x_offset = 50
    y_offset = 150
    l_img = result
    s_img = cv2.resize(out_img, None, fx=.2, fy=.2)
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], 0] = s_img[:,:,0]
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], 1] = s_img[:,:,1]
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], 2] = s_img[:,:,2]


    for c in range(0,2):
        l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] = s_img[:,:,c] * (s_img[:,:,2]/255.0) +  l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] * (1.0 - s_img[:,:,2]/255.0)

    return l_img
