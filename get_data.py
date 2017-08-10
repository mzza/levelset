import cv2

def get_image_data(path):
    image_data=cv2.imread(path,0)
    return image_data

if __name__=="__main__":
    image_matrix=get_image_data('/home/lza/Documents/level_set/lena.jpg')
    print(image_matrix)
