#
'''Based on;
[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

And

https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/
by Jason Brownlee on July 1, 2019 in Generative Adversarial Networks

'''
#

import os
from os import listdir
from os.path import isfile, join

import numpy as np
from numpy import ones
from numpy.random import randint

import math
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Dropout
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


#
# The discriminator
#
def build_discriminator(input_shape):

	kernelsize = (5,5)

	model = Sequential(name='discriminator')
	# normal
	model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=input_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128,kernel_size=kernelsize, strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, kernel_size=kernelsize, strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(256, kernel_size=kernelsize, strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(256, kernel_size=kernelsize, padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	
	return model

#
# The Generator
#
def build_generator(inputs, latent_size):

	kernelsize = (5,5)

	model = Sequential(name='generator')

	n_nodes = 256 * 24 * 24
	
	model.add(Dense(n_nodes, input_dim=latent_size))
	model.add(LeakyReLU(alpha=0.2))
	
	model.add(Reshape((24, 24, 256)))

	model.add(Conv2DTranspose(128, kernel_size=kernelsize, strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2DTranspose(128,kernel_size= kernelsize, strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2DTranspose(128,kernel_size= kernelsize, strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2DTranspose(256,kernel_size= kernelsize, strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2DTranspose(256,kernel_size= kernelsize, padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(3, kernel_size= kernelsize, activation='tanh', padding='same'))
	
	return model


#
# Model building
#
def build_models(image_size, latent_size, model_name):

	# Learning rate
	lr = 2e-4
	#  Learning rate decay
	decay = 6e-8
	
	# Input shape image
	input_shape = (image_size, image_size, 3)

	# build the discriminator
	inputs = Input(shape=input_shape, name='discriminator_input')
	discriminator = build_discriminator(input_shape)
	optimizer = RMSprop(lr=lr, decay=decay)
	discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	discriminator.trainable = False
	discriminator.summary()

	# build the generator
	input_shape = (latent_size,)
	inputs = Input(shape=input_shape, name='z_input')
	generator = build_generator(inputs, latent_size)
	generator.summary()

	# gan = generator + discriminator
	optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)
	gan = Model(inputs, discriminator(generator(inputs)), name=model_name)
	gan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	gan.summary()

	models = (generator, discriminator, gan)
	
	return models


#
# Training
#
def train(models, x_train, batch_size, latent_size, train_steps, save_interval, model_name):

	generator, discriminator, gan = models

	# noise vectors x 5, to see the development on 5 different latent spaces
	noise_input = []
	for x in range(5):
		noise_input.append(np.random.uniform(-1.0, 1.0, size=[1, latent_size]))

	train_size = x_train.shape[0]
	
	try:

		for i in range(train_steps):
			rand_indexes = np.random.randint(0, train_size, size=batch_size)
			real_images = x_train[rand_indexes]
	
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
		
			fake_images = generator.predict(noise)
		
			x = np.concatenate((real_images, fake_images))
		
			# Real is 1, Fake is 0
			y = np.ones([2 * batch_size, 1])
			y[batch_size:, :] = 0.0
		
			loss, acc = discriminator.train_on_batch(x, y)
			log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
			y = np.ones([batch_size, 1])
		
			loss, acc = gan.train_on_batch(noise, y)
			log = "%s [gan loss: %f, acc: %f]" % (log, loss, acc)
			print(log)
		
			if (i + 1) % save_interval == 0:
				for x in range(5):
					save_image(generator, noise_input=noise_input[x], show=False, name="image_" + str(i) + "_" + str(x), model_name=model_name)
   
	except KeyboardInterrupt:
		pass

	save_models(generator, discriminator, gan)
	exit()
	

def save_models(generator, discriminator, gan):
	
	generator.save(model_name + "_generator.h5")
	discriminator.save(model_name + "_discriminator.h5")
	gan.save(model_name + "_gan.h5")


# predict
def predict(generator,latent_size):

	generator = load_model(model_name + ".h5")
	noise_input = np.random.uniform(-1.0, 1.0, size=[1, latent_size])
	
	save_image(generator,noise_input=noise_input, show=True, model_name="predict_outputs")


# save image
def save_image(generator, noise_input, show=False, name="test", model_name="gan"):
	
	os.makedirs(model_name, exist_ok=True)
	filename = os.path.join(model_name, name + ".png")
	imagedata = generator.predict(noise_input)[0]
	image = Image.fromarray(np.uint8(imagedata)*255)
	image.save(filename,"png")

if __name__ == "__main__":

	predict = False

	model_name = "cheesecake_gan"
	image_src_dir = "./cheesecake/"
	
	latent_size = 100
	# Same for x,y for now
	image_size = 384

	if predict:
		predict(latent_size)
		
	else:
		#Increase batch size if more available memory or smaller image size
		batch_size = 24
		
		# CTRL-C will also save the models.
		train_steps = 35000
		save_interval = 500
	
		dircontent = listdir(image_src_dir)
		onlyimages = [f for f in dircontent if f.endswith(".jpg") ]

		x_train = np.empty([0,image_size,image_size,3])

		for f in onlyimages:
			image = Image.open(image_src_dir + f)
			image_array = np.asarray(image)
			#print(image_array.shape,f)
			x_train = np.append(x_train,[image_array],axis=0)
	
		x_train = np.reshape(x_train, [-1, image_size, image_size, 3])
		x_train = x_train.astype('float32') / 255
		
		print("loaded images : ", x_train.shape)
	
		print(x_train.shape)

		models = build_models(image_size, latent_size, model_name)

		train(models, x_train, batch_size, latent_size, train_steps, save_interval, model_name)

