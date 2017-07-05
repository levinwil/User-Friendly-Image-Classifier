'''
video

Description: captures timeseries pictures and saves them in the correct
directory according to the command line input 'mask'. Automatically splits
into testing and training data.

Author: Will LeVine
Email: will.levine0@gmail.com
Date: Jul 5, 2017
'''
import numpy as np
import cv2
import scipy.misc
import time
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser(description='A program to capture video \
and save timestep pictures along the way, splitting it into trainin and \
validaiton data along the way.')
parser.add_argument("--mask", help="If you are capturing data of someone \
wearing a mask, tye true after. Otherwise, type false after..")
args = parser.parse_args()

if args.mask == True:
    path_to_validate = "data/validation/Mask/"
    path_to_train = "data/train/Mask/"
    print "HERE"
else:
    path_to_validate = "data/validation/No_Mask/"
    path_to_train = "data/train/No_Mask/"

def extract_number(s):
    index_of_dot = s.index('.')
    s = s[7: index_of_dot]
    return int(s)

files = [extract_number(f) for f in listdir(path_to_train) if \
isfile(join(path_to_train, f))]

cap = cv2.VideoCapture(0)
count = np.max(files) + 10
print count
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('name', frame)
    count = count + 1
    time.sleep(.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count % 5 == 1:
        cv2.imwrite(path_to_validate + "picture" + str(count) + ".jpg", frame)
    else:
        cv2.imwrite(path_to_train + "picture" + str(count) + ".jpg", frame)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
