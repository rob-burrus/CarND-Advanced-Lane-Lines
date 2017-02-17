import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from image_processing import process_image
from line_tracker import line_tracker

line_tracker = line_tracker()


def process(img):
    processed = process_image(img, line_tracker)
    return processed

output_video = 'output_t_video.mp4'
input_video = 'project_video.mp4'

clip = VideoFileClip(input_video)
video_clip = clip.fl_image(process)
video_clip.write_videofile(output_video, audio=False)
