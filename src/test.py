import numpy as np
from skimage import io, color
from skimage.transform import resize
from keras.models import load_model
import pdb

if __name__ == '__main__':

	model = load_model('generator.h5')

	img = io.imread('dog.jpg')
	img = np.array(img, dtype = float)/255.0
	resized_img = resize(img, (32, 32))
	gray = color.rgb2gray(resized_img)
	gray = np.expand_dims(gray, axis = -1)
	gray = np.expand_dims(gray, axis = 0)

	out = model.predict(gray)

	out[:,:,:,0] = (out[:,:,:,0] + 1)*50.0
	out[:,:,:,1] = (out[:,:,:,1])*110.0
	out[:,:,:,2] = (out[:,:,:,2])*110.0

	out = np.array(out, dtype = float)

	rgb = color.lab2rgb(out[0])

	io.imsave("color_img.png", rgb)
