import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture('train.mp4')

#for frame identity
def extractImages(nFrames):

    # Creates new file for extracted images from video
    if not os.path.exists('images'):
        os.makedirs('images')

    index = 0
    while(index < nFrames):

        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Saves images
        name = './images/frame' + str(index) + '.jpg'
        cv.imwrite(name, gray)

        # next frame
        index += 1

def newTrainImages(space, index, nFrames):

    if not os.path.exists('train_images'):
        os.makedirs('train_images')

    while index<nFrames:
        img1_name = './images/frame' + str(index-space) + '.jpg'
        img2_name = './images/frame' + str(index+space) + '.jpg'
        img1 = cv.imread(img1_name, 0)
        img2 = cv.imread(img2_name, 0)

        # Subtracting both images
        diff = cv.absdiff(img1, img2)

        # Set every pixel that changed by 40 to 255, and all the others to zero
        thershold_value = 40
        set_to_value = 255
        ret, thresh = cv.threshold(diff, thershold_value, set_to_value, cv.THRESH_BINARY)

        # Saves images
        diff_name = './train_images' + '/frame_' + str(index-space) + '_' + str(index+space) + '.jpg'
        print ('Creating...' + diff_name)
        cv.imwrite(diff_name, thresh)

        # next frame subtraction
        index += 2*space

        if index>nFrames:
            break

def getLabels(space, index, nFrames):
    
    with open("train.txt") as train:
        with open("target.txt","w") as target:
            for line in train.read().split("\n")[space::2*space]:
                target.write(line)
                target.write("\n")
                index += 2*space
                if index+2*space>nFrames:
                    break

nFrames = 20400
space = 3
index = 3

extractImages(nFrames)
newTrainImages(space, index, nFrames)
getLabels(space, index, nFrames)

cap.release()
cv.destroyAllWindows()