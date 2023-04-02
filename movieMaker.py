import cv2
import numpy as np
import glob
import os
frameSize = (500, 500)

# out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60, frameSize)

# for filename in glob.glob('/Users/georgienahass/Desktop/images_control_test/'):
#     img = cv2.imread(filename)
#     out.write(img)

# out.release()

image_folder = 'images'
video_name = 'video.mp4'

images = [img for img in os.listdir('/Users/georgienahass/Desktop/images_control_test/')]
print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()