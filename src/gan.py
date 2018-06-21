import numpy as np
from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Conv2DTranspose, Concatenate, Dropout, Dense, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras import optimizers
from skimage import io, color
import h5py
import pdb
import cPickle
import os

class GAN():

	def __init__(self):

		self.nrows = 32
		self.ncols = 32
		self.nchannels = 1

		self.discriminator = self.build_discriminator()
		self.generator = self.build_generator()

		self.discriminator.compile(optimizer=optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
		self.generator.compile(optimizer=optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='mean_squared_error')

	def build_discriminator(self):

		inputs = Input(shape = (self.nrows, self.ncols, 4))

		encoder = Conv2D(64, (4, 4), strides = (2, 2), activation = None, padding = "same")(inputs)
		encoder = BatchNormalization()(encoder)
		encoder = LeakyReLU(0.2)(encoder)

		encoder1 = Conv2D(128, (4, 4), strides = (2, 2), activation = None, padding = "same")(encoder)
		encoder1 = BatchNormalization()(encoder1)
		encoder1 = LeakyReLU(0.2)(encoder1)

		encoder2 = Conv2D(256, (4, 4), strides = (2, 2), activation = None, padding = "same")(encoder1)
		encoder2= BatchNormalization()(encoder2)
		encoder2 = LeakyReLU(0.2)(encoder2)

		encoder3 = Conv2D(512, (4, 4), strides = (1, 1), activation = None, padding = "same")(encoder2)
		encoder3 = BatchNormalization()(encoder3)
		encoder3 = LeakyReLU(0.2)(encoder3)

		encoder3 = Flatten()(encoder3)

		out = Dense(1, activation = 'sigmoid')(encoder3)

		return Model(inputs = inputs, outputs = out)


	def build_generator(self):

		inputs = Input(shape = (self.nrows, self.ncols, self.nchannels))

		encoder = Conv2D(64, (4, 4), strides = (1, 1), activation = None, padding = "same")(inputs)
		encoder = BatchNormalization()(encoder)
		encoder = LeakyReLU(0.2)(encoder)

		encoder0 = Conv2D(64, (4, 4), strides = (1, 1), activation = None, padding = "same")(encoder)		#32
		encoder0 = BatchNormalization()(encoder0)
		encoder0 = LeakyReLU(0.2)(encoder0)

		encoder1 = Conv2D(128, (4, 4), strides = (2, 2), activation = None, padding = "same")(encoder0)		#16
		encoder1 = BatchNormalization()(encoder1)
		encoder1 = LeakyReLU(0.2)(encoder1)

		encoder2 = Conv2D(256, (4, 4), strides = (2, 2), activation = None, padding = "same")(encoder1)		#8
		encoder2= BatchNormalization()(encoder2)
		encoder2 = LeakyReLU(0.2)(encoder2)

		encoder3 = Conv2D(512, (4, 4), strides = (2, 2), activation = None, padding = "same")(encoder2)		#4
		encoder3 = BatchNormalization()(encoder3)
		encoder3 = LeakyReLU(0.2)(encoder3)

		encoder4 = Conv2D(1024, (4, 4), strides = (2, 2), activation = None, padding = "same")(encoder3)	#2
		encoder4 = BatchNormalization()(encoder4)
		encoder4 = LeakyReLU(0.2)(encoder4)

		decoder4 = Conv2DTranspose(512, (4,4), strides = (2, 2), activation = 'relu', padding = "same")(encoder4)	#4
		decoder4 = Concatenate(axis = -1)([encoder3, decoder4])
		decoder4 = BatchNormalization()(decoder4)
		decoder4 = Dropout(0.8)(decoder4)

		decoder3 = Conv2DTranspose(256, (4,4), strides = (2, 2), activation = 'relu', padding = "same")(decoder4)	#8
		decoder3 = Concatenate(axis = -1)([encoder2, decoder3])
		decoder3 = BatchNormalization()(decoder3)
		decoder3 = Dropout(0.8)(decoder3)

		decoder2 = Conv2DTranspose(128, (4,4), strides = (2, 2), activation = 'relu', padding = "same")(decoder3)	#16
		decoder2 = Concatenate(axis = -1)([encoder1, decoder2])
		decoder2 = BatchNormalization()(decoder2)
		decoder2 = Dropout(0.8)(decoder2)

		decoder1 = Conv2DTranspose(64, (4,4), strides = (2, 2), activation = 'relu', padding = "same")(decoder2)	#32
                decoder1 = Concatenate(axis = -1)([encoder0, decoder1])
                decoder1 = BatchNormalization()(decoder1)
		
		decoder = Conv2DTranspose(64, (4,4), strides = (1, 1), activation = 'relu', padding = "same")(decoder1)		#32
		decoder = Concatenate(axis = -1)([encoder, decoder])
		decoder = BatchNormalization()(decoder)

		decoder = Conv2DTranspose(3, (1,1), strides = (1, 1), activation = 'tanh', padding = "same")(decoder)
		
		return Model(inputs = inputs, outputs = decoder)


	def train(self, epochs = 50, batch_size = 32, data = [], lab_ = []):


		for n in range(epochs):

			for idx in range(0, data.shape[0], batch_size):

				#idx = np.random.randint(0, data.shape[0], batch_size)
				batch = data[idx:idx+batch_size]

				lab = lab_[idx:idx+batch_size]
				gray = color.rgb2gray(batch)
				gray = np.expand_dims(gray, axis = -1)

				fake = np.zeros((batch.shape[0], 1))
				real = np.ones((batch.shape[0], 1))

				real_inp = np.concatenate((gray, lab), axis = -1)

				fake_pred = self.generator.predict(gray)

				fake_inp = np.concatenate((gray, fake_pred), axis = -1)

				d_loss_1 = self.discriminator.train_on_batch(real_inp, real)
				d_loss_2 = self.discriminator.train_on_batch(fake_inp, fake)

				d_loss = (d_loss_1[0] + d_loss_2[0])/2.0
				accuracy = (d_loss_1[1] + d_loss_2[1])/2.0

				gen_loss = self.generator.train_on_batch(gray, lab)

			print "Epoch: {}".format(n)
			print "Discriminator loss: {}, Accuracy: {}".format(d_loss, accuracy)
			print "Generator loss: {}".format(gen_loss)

		self.discriminator.save('discriminator.h5')
		self.generator.save('generator.h5')


if __name__ == '__main__':


	directory = '../data/cifar-10/cifar-10-batches-py'
	

	for i in range(1,6):
		filename = "data_batch_%d"%i
		filename = os.path.join(directory, filename)
		with open(filename, 'rb') as fo:
			dict = cPickle.load(fo)
			if i == 1:
				data = dict['data']
			else:
				data = np.concatenate((data, dict['data']), axis = 0)


	data = np.array(data)
	data = np.array(data, dtype=float) / 255.0
	data = data.reshape(len(data), 3, 32, 32).transpose(0,2,3,1)
	lab = color.rgb2lab(data)
	lab[:,:,:,0] = lab[:,:,:,0]/50.0 - 1
	lab[:,:,:,1] = lab[:,:,:,1]/110.0
	lab[:,:,:,2] = lab[:,:,:,2]/110.0


	gan = GAN()
	gan.train(epochs = 10, batch_size = 32, data = data, lab_ = lab)

