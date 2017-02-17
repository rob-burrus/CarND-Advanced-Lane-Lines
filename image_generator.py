import cv2
import glob
import matplotlib.pyplot as plt
from image_processing import process_image
from line_tracker import line_tracker

images = glob.glob('./video_images/test*.jpg')
line_tracker = line_tracker()

for idx, fname in enumerate(images):

    img = cv2.imread(fname)
    result = process_image(img, line_tracker)


    #f, (img1, img2) = plt.subplots(1, 2)
    #f.tight_layout()
    #img1.imshow(combined_binary, cmap='gray')
    #img2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    #plt.imshow(result)

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


    #write_name = './test_images/tracked' + str(idx) + '.jpg'
    #cv2.imwrite(write_name, result)
