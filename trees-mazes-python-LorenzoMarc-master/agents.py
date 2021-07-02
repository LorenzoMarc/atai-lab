import cv2
import random
import numpy as np

WALL = np.array([0, 0, 0])
def setagent(path):
    img = cv2.imread(path) # read an image from a file using
    number_of_white_pix = np.sum(img == 255)
    where = np.argwhere(img == 255)
    random_position = random.randrange(len(where))
    redagent = where[random_position][:-1]
    cv2.circle(img, (1,0), 0, (255,0,0), -1) #BGR
    cv2.circle(img, (redagent[1], redagent[0]), 0, (0, 0, 255), -1)  # BGR
    print((redagent[1], redagent[0]))
    cv2.imwrite(path,img)