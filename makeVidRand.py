import cv2
import numpy as np
from PIL import Image
import os



for i in range(0,500):
    arr = np.random.randint(255, size=(400, 400), dtype=np.uint8)    
    im = Image.fromarray(arr)
    
    # Converting the numpy array into image Saving the image
    name = str(i) +'.png'
    im.save('/Users/georgienahass/Desktop/images_control_test/'+name)
    
    
video_name = 'randomVidLonger.avi'
image_folder = '/Users/georgienahass/Desktop/images_control_test/'

images = [img for img in os.listdir('/Users/georgienahass/Desktop/images_control_test/')]

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 20, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

