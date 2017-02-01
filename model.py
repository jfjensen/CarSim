import csv
import numpy as np 
import tensorflow as tf
import cv2

import os
import argparse
import json

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ParametricSoftplus
from keras.optimizers import Adam
from keras.regularizers import l2, activity_l2
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
# from keras.utils.visualize_util import plot

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.misc import imread, imresize

# read in the images

data_path = "./simulator-windows-64/data/driving_log.csv"
with open(data_path, 'r') as data_file:
	reader = csv.reader(data_file)
	data = np.array([row for row in reader])
print(data.shape)
data = data[1:]

img_data_col = 0
steering_data_col = 3

data_features, data_labels = np.reshape(data[:,0:3],(len(data),-1)), np.reshape(data[:,steering_data_col], (len(data),-1))
print(data_features.shape)
print(data_labels.shape)

data_features, data_labels = shuffle(data_features, data_labels)

train_features, valid_features, train_labels, valid_labels = train_test_split(data_features, data_labels, test_size=0.2, random_state=42)

# global variables for size of camera image to feed to neural network
ROWS, COLS = 80, 320

def preprocess(img):
	# crop the camera image to remove what's above the road and also remove the hood of the car
	img = img[55:135,:,:]
	return img


def read_img(img_path):
	# fetch the camera image given a file name
	img = imread("C:/Users/JFJ/Documents/GitHub/CarND/CarSim/simulator-windows-64/data/"+img_path.strip())
	return img

def get_augmented(features, labels, idxs):
	aug_features = []
	aug_labels = []
	
	# for every image in the list of indices
	for ix in idxs:
		
		# get the steering angle from the training data
		steering = labels[ix].astype(float)

		# choose center/left/right camera image at random
		img_choice = np.random.choice([0,1,2])

		# fetch and pre-process the chosen camera image (crop it)
		img = preprocess(read_img(features[ix][img_choice]))
	
		# adjust the steering angle if camera image from left or right camera
		steering_angle = 0.25
		if img_choice == 1:
			steering = steering + steering_angle
		elif img_choice == 2:
			steering = steering - steering_angle

		if abs(steering) > 0.01:
			# shift the camera image left or right according to normal distribution			
			max_shift = 100
			shift_dist = 0.004
			steering_shift = (np.random.normal(0.0,0.25)*max_shift)
			M = np.float32([[1,0,steering_shift],[0,1,0]])
			img = cv2.warpAffine(img,M,(COLS,ROWS))
			steering = steering + (steering_shift * shift_dist)

		# creating random brightness changes (needed for track 2)
		brightness_correction = 0.2
		hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		hsv[:,:,2] = hsv[:,:,2] * (brightness_correction + np.random.uniform())
		img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

		# flip the camera image left/right in order to remove the bias towards left turns
		if np.random.random() > 0.5:
			steering = -1.0 * steering
			img = cv2.flip(img, 1)

		aug_features.append(img)
		aug_labels.append(steering)

	return np.array(aug_features), np.array(aug_labels)


def generate(features, labels, batch_size):
    # generating batches for keras using an infinite loop
    while True:
    	# create a list of random indices
        idxs = np.random.choice(len(features), batch_size)

        # augment the images given the list of indices
        batch_features, batch_labels = get_augmented(features, labels, idxs)

        yield batch_features, batch_labels


# create and train the model
# adapted from Comma.ai - https://github.com/commaai/research/blob/master/train_steering_model.py
def get_model(time_len=1, learning_rate=0.001):
	
	ch, row, col = 3, ROWS, COLS  # pre-processed format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
	        input_shape=(row, col, ch),
	        output_shape=(row, col, ch)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(MaxPooling2D())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(1024))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1, activation='tanh'))
	
	# set the learning rate for the Adam optimizer...
	model.compile(optimizer=Adam(lr=learning_rate), loss="mse")

	return model


if __name__ == "__main__":
	
	# setting some variables which can be over-ridden via command line input
	parser = argparse.ArgumentParser(description='Steering angle model trainer')
	parser.add_argument('--rate', type=float, default=0.001, help='Learning rate.')
	parser.add_argument('--batch', type=int, default=128, help='Batch size.')
	parser.add_argument('--epoch', type=int, default=5, help='Number of epochs.')
	parser.add_argument('--epochsize', type=int, default=20000, help='How many frames per epoch.')
	parser.add_argument('--validsize', type=int, default=3000, help='How many validation samples.')
	args = parser.parse_args()

	train_features, train_labels = shuffle(train_features, train_labels)
	samples_per_epoch = (args.epochsize//args.batch)*args.batch

	# fetch the model of the convolutional neural network
	model = get_model(learning_rate=args.rate)

	# train the model
	model.fit_generator(
	    generator=generate(train_features, train_labels, args.batch),
	    samples_per_epoch=samples_per_epoch,#args.epochsize,
	    nb_epoch=args.epoch,
	    validation_data=generate(valid_features, valid_labels, args.batch),
	    nb_val_samples=args.validsize
	)

	# saving the model for use with the drive.py script
	json = model.to_json()
	model.save_weights('./model.h5')
	with open('./model.json', 'w') as out:
		out.write(json)

	