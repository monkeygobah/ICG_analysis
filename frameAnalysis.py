from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
(major, minor) = cv2.__version__.split(".")[:2]
import tkinter as tk     # from tkinter import Tk for Python 3.x
import tkinter.filedialog as fd # askopenfilename


""" Algorithm to detect average intensity of every frame in a video. 
Every frame is converted to gray scale, so this algorithm is best suited for 
black and white surgical videos as opposed to full color vids. Modification to the 
averaging process will need to be implemented to accurately accomodate full color vids.

At the bottom of the algorithm, we use the logistic function to try and fit a a sigmoidal curve
to the data, as this is the theoretical shape of the observed intensity graph. This output can be removed if desired.
"""

# Prompt user to select file from system
# root = tk.Tk()
# files = fd.askopenfilenames(parent=root, title='Choose a file')
# file_list = list(files)


# file_list = ['/Users/georgienahass/Desktop/ocpVID1.mp4']

def fileSelect():
    root = tk.Tk()
    files = fd.askopenfilenames(parent=root, title='Choose a file')
    file_list = list(files)
    print(file_list)
    return (file_list)

file_list = fileSelect()
print(file_list)

# If multiple vids selected, iterate through all of them 

for vid in file_list:
    # Otherwise, grab a reference to the video file
    vs = cv2.VideoCapture(vid)
    fps = vs.get(cv2.CAP_PROP_FPS)
    fps1 = int(fps)

    intensity_out_test = []
    time_tick = []
    t = 0
    # loop over frames from the video stream
    while True:
        # Grab the current frame, then handle if we are using a VideoStream or VideoCapture object
        ret, frame = vs.read()

        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    # Resize the frame (so we can process it faster) and grab the frame dimensions
        frame = imutils.resize(frame, width=1000)

        (H, W) = frame.shape[:2]
        #get total number of pixels from frame
        pixel_count = H * W
        intensity_test = np.mean(frame)
        # This division process gives you the average intensity. Not mathematically 
        # necessary when working with the whole frame, but it is a more standard way to plot.
        # intensity_test = intensity_test / pixel_count
        
        # Check if data is NaN or if there are any runtime warnings. If either of these are true, do not keep the value and append 0.
        # This loop is the main data output loop as intensity out test is what ends up getting plotted
        try: 
            if math.isnan(intensity_test):
                intensity_out_test.append(0)
            else:    
                intensity_out_test.append(intensity_test)
        except RuntimeWarning:
            intensity_out_test.append(0)
                
        time_tick.append(t)
        t += 1

        # Show the output frame- turning on results in speed decrease

 
 
    # cv2.destroyAllWindows()

    # Use the fps of video to make sure that length of data out is divisible by FPS. Make time value reflect this as well
    while len(intensity_out_test) % fps1 != 0:
        intensity_out_test.pop(-1)
        time_tick.pop(-1)

    # Convert data to np array. There is def a better way to do this but lazy
    avgTestArr = np.array(intensity_out_test)
    avgTimeArr = np.array(time_tick)

    # Make new array with every elemennt being the average of every n elements (n = fps of video). 
    # Reduces noise and allows for frame -> second conversion
    avgTestAverage = np.average(avgTestArr.reshape(-1, fps1), axis=1)
    avgTime = np.average(avgTimeArr.reshape(-1, fps1), axis=1)
    avgTestAverageList = avgTestAverage.tolist()
    # Make time axis make sense
    timeAxis = []
    for y in range(len(avgTime)):
        timeAxis.append(y)

    # Equation for the logistic fit curve
    def sigmoid(x, L ,x0, k, b):
        return L / (1 + np.exp(-k*(x-x0))) + b

    # Provide an initial guess for the loigistic fit operation 
    p0 = [max(avgTestAverage.tolist()), np.median(timeAxis),1,min(avgTestAverageList)]

    # popt contains all of the variables from the logistic fit. Use * to expand the return 
    # Value of popt and calucalte the y values on the next line
    popt, pcov = curve_fit(sigmoid, timeAxis, avgTestAverageList,p0, method='dogbox', maxfev=10000)
    y_fit = sigmoid(timeAxis, *popt)
    
    
    
    """        
        now begins the next phase of the analysis- thresholding and counting important stuff
    """
        
    #### started editing here 01/18/23
    minInflect = max(idx for idx, val in enumerate(y_fit) if val == min(y_fit))
    maxInflect = min(idx for idx, val in enumerate(y_fit) if val == max(y_fit))
    print(minInflect)
    print(maxInflect)
    timeToPeak = y_fit[minInflect]
    ave50Fluor = (max(y_fit) + min(y_fit)) / 2
    maxFluorTime = timeAxis[avgTestAverageList.index(max(avgTestAverageList))]

    print('time to peak', round(timeToPeak,2))
    print('ave 50', round(ave50Fluor,2))
    print('time of max fluor', maxFluorTime)

    # reopen video and use new information to calculate number of pixels in each frame with intensity greater than Ave50
    vs1 = cv2.VideoCapture(vid)
    white_list = []
    while True:
        # Grab the current frame, then handle if we are using a VideoStream or VideoCapture object
        ret, frame = vs1.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    # Resize the frame (so we can process it faster) and grab the frame dimensions
        frame = imutils.resize(frame, width=1000)
        (H, W) = frame.shape[:2]
        #get total number of pixels from frame
        pixel_count = H * W
        intensity_test = np.mean(frame)
        # This division process gives you the average intensity. Not mathematically 
        # necessary when working with the whole frame, but it is a more standard way to plot.
        # intensity_test = intensity_test / pixel_count
        
        # Check if data is NaN or if there are any runtime warnings. If either of these are true, do not keep the value and append 0.
        # This loop is the main data output loop as intensity out test is what ends up getting plotted
        try: 
            ret,thresh1 = cv2.threshold(frame,ave50Fluor,255,cv2.THRESH_BINARY)
            # cv2.imshow("Frame", thresh1)
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord("q"):
            #         break
            # Count white pixels
        
            white  = np.count_nonzero(thresh1)
            white_list.append(white)
            # print(f"white: {white}")
                    
        except RuntimeWarning:
            intensity_out_test.append(0)
                
    while len(white_list) % fps1 != 0:
        white_list.pop(-1)
    whiteArr = np.array(white_list)
    avgWhite = np.average(whiteArr.reshape(-1, fps1), axis=1)


    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))

    # Plot data on the first subplot
    ax1.plot(timeAxis, avgTestAverageList, color='black', label = 'Data')
    ax1.plot(timeAxis, y_fit, label = 'fit')
    ax1.set_ylabel('Average intensity (arbu)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_title("Intensity Analysis")
    ax1.legend()

    # Plot data on the second subplot
    ax2.plot(timeAxis, avgWhite.tolist(), color='black')
    ax2.set_title("Area of Fluorescence")
    ax2.set_ylabel('Pixels Above Threshold')
    ax2.set_xlabel('Time (seconds)')


    # Plot everything 
    # plt.plot(timeAxis, avgTestAverageList, color='black', label = 'Data')
    # plt.plot(timeAxis, y_fit, '-', label = 'fit')
    # plt.title('Intensity Analysis')
    # plt.xlabel('Seconds')
    # plt.legend()
    # ax = plt.gca()
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([0, 255])

    # plt.ylabel('Average intensity (arbu)')
    plt.tight_layout()

    plt.show()
    # plt.savefig('oculoVid1_box')


