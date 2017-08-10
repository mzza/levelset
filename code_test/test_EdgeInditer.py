import len_term
import get_data
import numpy as np
import matplotlib.pyplot  as plt
import cv2
#import matplotlib.image as mpimg

if __name__ == '__main__':
    image_matrix=get_data.get_image_data('/home/lza/Documents/level_set/gourd.bmp')
    g=len_term.g_inditer(image_matrix)
    cv2.imshow("'g",g)
    k=cv2.waitKey(0)
    if k==27:
    	cv2.destroyAllWindow()