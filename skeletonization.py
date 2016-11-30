import cv2
import numpy as np
import imutils


def skeleton(self,frames):

    image = cv2.imread(frames,1)
    cv2.imshow('image',image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    skeleton = imutils.skeletonize(gray, size=(3, 3))

    cv2.imshow("Skeleton", skeleton)
    cv2.imwrite('skeleton[i].jpg',skeleton)

    k = cv2.waitKey(0)

    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('messigray.png',image)
        cv2.destroyAllWindows()

