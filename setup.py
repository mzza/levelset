import regularization
import get_data

image_matrix=get_data.get_image_data('/home/lza/Documents/level_set/image/gourd.bmp')
clf=regularization.levelset(2,0.2,1,'random',200,0,1.5,0)
clf.fit(image_matrix)
labels=clf.predict()

