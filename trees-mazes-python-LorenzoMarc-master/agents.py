import cv2
import matplotlib.pyplot as plt
import numpy as np

def setagent(path, size):
    img = cv2.imread(path) # read an image from a file using
    cv2.circle(img, size, 1, (0,0,255), -1) #BGR
    cv2.circle(img, (1,0), 1, (255,0,0), -1) #BGR
    cv2.imwrite(path,img)