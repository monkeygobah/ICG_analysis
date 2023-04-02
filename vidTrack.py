from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2
import numpy as np
vid = './gold_weight.mp4'
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# import Tkinter,tkFileDialog

import math
(major, minor) = cv2.__version__.split(".")[:2]


OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    # "kcf": cv2.TrackerKCF_create,
    # "boosting": cv2.TrackerBoosting_create,
    # "mil": cv2.TrackerMIL_create,
    # "tld": cv2.TrackerTLD_create,
    # "medianflow": cv2.TrackerMedianFlow_create,
    # "mosse": cv2.TrackerMOSSE_create
}
 
tracker = OPENCV_OBJECT_TRACKERS['csrt']()


# initialize the bounding box coordinates of the object we are going to track
initBBTest = None
initBBControl = None

# otherwise, grab a reference to the video file
vs = cv2.VideoCapture(vid)
fps = vs.get(cv2.CAP_PROP_FPS)
fps1 = int(fps)
# initialize the FPS throughput estimator

r_tick_t = []
g_tick_t = []
b_tick_t = []
r_tick_c = []
g_tick_c = []
b_tick_c = []

intensity_out_test = []
intensity_out_control = []
print('made')
time_tick = []
t = 0
# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a VideoStream or VideoCapture object
    ret, frame = vs.read()
    # frame = frame[1] #if args.get("video", False) else frame check to see if we have reached the end of the stream
    if frame is None:
        break
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# resize the frame (so we can process it faster) and grab the frame dimensions
    frame = imutils.resize(frame, width=1000)
    (H, W) = frame.shape[:2]
	# check to see if we are currently tracking an object
    if initBBTest is not None and initBBControl is not None:
		# grab the new bounding box coordinates of the objectWAR
        (success, box) = tracker.update(frame)
		# check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + static_w_test, y + static_h_test), (0, 255, 0), 3)
            region = frame[yt:yt+static_h_test, xt:xt+static_w_test]
            intensity_test = np.mean(region)
        
            bounding_box_pixels_TEST = (xt + static_w_test) * (yt + static_h_test)
            intensity_test = intensity_test / bounding_box_pixels_TEST
            
            try: 
                if math.isnan(intensity_test):
                    intensity_out_test.append(0)
                else:    
                    intensity_out_test.append(intensity_test)
            except RuntimeWarning:
                intensity_out_test.append(0)
            
            # (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (xc, yc), (xc + static_w_cont, yc + static_h_cont), (0, 255, 0), 3)
            region = frame[yc:yc+static_h_cont, xc:xc+static_w_cont]
            intensity_control = np.mean(region)            
            
            bounding_box_pixels_CONTROL = (xc + static_w_cont) * (yc + static_h_cont)
            intensity_control = intensity_control / bounding_box_pixels_CONTROL
            
            try: 
                if math.isnan(intensity_control):
                    intensity_out_control.append(0)
                else:    
                    intensity_out_control.append(intensity_control)
            except RuntimeWarning:
                intensity_out_control.append(0)
                
            time_tick.append(t)
            t += 1
    
		# update the FPS counter

	
	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the 's' key is selected, we are going to "select" a bounding box to track
    if key == ord("t"):
    # if t == 0:
    		# select the bounding box of the object we want to track (make sure you press ENTER or SPACE after selecting the ROI)
        initBBTest = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        xt,yt,static_w_test,static_h_test = initBBTest
		# start OpenCV object tracker using the supplied bounding box coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBBTest)
        
    	# select the bounding box of the object we want to track (make sure you press ENTER or SPACE after selecting the ROI)
        initBBControl = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        xc,yc,static_w_cont,static_h_cont = initBBControl
		# start OpenCV object tracker using the supplied bounding box coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBBControl)
        
	# if the `q` key was pressed, break from the loop
    elif key == ord("q"):
	    break
 
 
# def objective (x, L, x0, k, b):
#     y = L / (1 + np.exp(-k*(x-x0))) + b
#     return y


average_control = sum(intensity_out_control) / len(intensity_out_control)
divided_intensities = [x / average_control for x in intensity_out_test]  

norm_values = [intensity_out_test[i]-intensity_out_control[i] for i in range(min(len(intensity_out_test), len(intensity_out_control)))]


while len(intensity_out_test) % fps1 != 0:
    intensity_out_test.pop(-1)
    intensity_out_control.pop(-1)
    norm_values.pop(-1)
    time_tick.pop(-1)


#convert everything to np arrays for averaging by fps 
avgTestArr = np.array(intensity_out_test)
avgControlArr = np.array(intensity_out_control)
avgTimeArr = np.array(time_tick)
avgNormArr = np.array(norm_values)

#average by fps values
avgTestAverage = np.average(avgTestArr.reshape(-1, fps1), axis=1)
avgControl = np.average(avgControlArr.reshape(-1, fps1), axis=1)
avgTime = np.average(avgTimeArr.reshape(-1, fps1), axis=1)
avgNorm = np.average(avgNormArr.reshape(-1, fps1), axis=1)

print((avgNorm))
print((avgTestAverage))

timeAxis = []
for y in range(len(avgTime)):
    timeAxis.append(y)
 
# # Equation for the logistic fit curve
# def sigmoid(x, L ,x0, k, b):
#     return L / (1 + np.exp(-k*(x-x0))) + b

# # Provide an initial guess for the loigistic fit operation 
# p0 = [max(avgNorm.tolist()), np.median(timeAxis),1,min(avgNorm.tolist())]

# # popt contains all of the variables from the logistic fit. Use * to expand the return 
# # Value of popt and calucalte the y values on the next line
# popt, pcov = curve_fit(sigmoid, timeAxis, avgNorm.tolist(),p0, method='dogbox')
# y_fit = sigmoid(timeAxis, *popt)


plt.plot(timeAxis, avgTestAverage.tolist(), color='black', label= 'Raw Test')
plt.plot(timeAxis, avgNorm.tolist(), color='green', label= 'Normalized')
# plt.plot(timeAxis, y_fit, '-', label = 'Normalized Fit')

plt.plot(timeAxis, avgControl.tolist(), color='red', label= 'Control')
plt.legend()
plt.title('Intensity Analysis')
plt.xlabel('Seconds')
plt.ylabel('Average intensity of ROI (arbu)')
plt.show()
plt.savefig('oculoVid1_box')

cv2.destroyAllWindows()