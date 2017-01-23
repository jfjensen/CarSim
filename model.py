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
# print(data[0,:])

img_data_col = 0
steering_data_col = 3
# data_features, data_labels = np.reshape(data[:,img_data_col],(len(data),-1)), np.reshape(data[:,steering_data_col], (len(data),-1))
data_features, data_labels = np.reshape(data[:,0:3],(len(data),-1)), np.reshape(data[:,steering_data_col], (len(data),-1))
print(data_features.shape)
print(data_labels.shape)

# print(data_features[0])
# print(data_labels[0])

data_features, data_labels = shuffle(data_features, data_labels)

train_features, valid_features, train_labels, valid_labels = train_test_split(data_features, data_labels, test_size=0.2, random_state=42)

ROWS, COLS = 80, 320


def preprocess(img):
	img = img[55:135,:,:]
	# img = cv2.resize(img,(COLS,ROWS))
	return img


# def read_imgs(img_paths):
# 	imgs = np.empty([len(img_paths), 160, 320, 3])
# 	# print("****************", img_paths[0])
# 	for i, path in enumerate(img_paths):
# 	    imgs[i] = imread("C:/Users/JFJ/Documents/GitHub/CarND/CarSim/simulator-windows-64/data/"+path[0])
# 	    # imgs[i] = imread(path[0])

# 	return imgs

def read_img(img_path):
	img = imread("C:/Users/JFJ/Documents/GitHub/CarND/CarSim/simulator-windows-64/data/"+img_path.strip())
	return img

def get_augmented(features, labels, idxs):
	aug_features = []
	aug_labels = []
	
	for ix in idxs:
		# img = read_img(features[ix])

		steering = labels[ix].astype(float)

		img_choice = np.random.choice([0,1,2])

		if img_choice == 1:
			steering = steering + 0.25
		elif img_choice == 2:
			steering = steering - 0.25


		img = preprocess(read_img(features[ix][img_choice]))
		

		if np.random.random() > 0.5:

			steering = -1.0 * steering
			img = cv2.flip(img, 1)

		aug_features.append(img)
		aug_labels.append(steering)

	return np.array(aug_features), np.array(aug_labels)


def generate(features, labels, batch_size):
    while True:
        idxs = np.random.choice(len(features), batch_size)
        # batch_features, batch_labels = read_imgs(features[idxs]), labels[idxs].astype(float)
        batch_features, batch_labels = get_augmented(features, labels, idxs)
        yield batch_features, batch_labels


# create and train the model
# adapted from Comma.ai - https://github.com/commaai/research/blob/master/train_steering_model.py
def get_model(time_len=1, learning_rate=0.001):
	# ch, row, col = 3, 160, 320  # camera format
	ch, row, col = 3, ROWS, COLS  # pre-processed format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
	        input_shape=(row, col, ch),
	        output_shape=(row, col, ch)))
	# model.add(Lambda(lambda x: x/127.5 - 1.,
	#         input_shape=(ch, row, col),
	#         output_shape=(ch, row, col)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))#, activation='tanh'))
	model.add(ELU())
	model.add(MaxPooling2D())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))#, activation='tanh'))
	model.add(ELU())
	# model.add(MaxPooling2D())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))#, activation='tanh'))
	# model.add(ELU())
	# model.add(MaxPooling2D())
	# model.add(Convolution2D(128, 5, 5, subsample=(2, 2), border_mode="same"))#, activation='tanh'))
	# model.add(ELU())
	# model.add(Convolution2D(256, 5, 5, subsample=(2, 2), border_mode="same"))#, activation='tanh'))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(1024, activation='tanh'))#, W_regularizer=l2(0.001)))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(512, activation='tanh'))#, W_regularizer=l2(0.001)))
	model.add(Dropout(.5))
	# model.add(ELU())
	# model.add(Dense(128, activation='tanh'))#, W_regularizer=l2(0.001)))
	# model.add(Dropout(.2))
	# model.add(ELU())
	# model.add(Dense(64, activation='tanh'))#, W_regularizer=l2(0.001)))
	# model.add(Dropout(.1))
	# model.add(Dense(32, activation='tanh'))#, W_regularizer=l2(0.001)))
	# model.add(Dropout(.1))
	# model.add(Dense(16, activation='tanh'))#, W_regularizer=l2(0.001)))
	# model.add(Dropout(.1))
	# model.add(ELU())
	model.add(Dense(1, activation='tanh'))
	
	# set the learning rate for the Adam optimizer...
	model.compile(optimizer=Adam(lr=learning_rate), loss="mse")

	return model


# from Comma.ai - https://github.com/commaai/research/blob/master/train_steering_model.py
def get_comma_model(time_len=1, learning_rate=0.001):
	ch, row, col = 3, ROWS, COLS  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
	        input_shape=(row, col, ch),
	        output_shape=(row, col, ch)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	# set the learning rate for the Adam optimizer...
	model.compile(optimizer=Adam(lr=learning_rate), loss="mse")

	return model



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Steering angle model trainer')
	parser.add_argument('--rate', type=float, default=0.001, help='Learning rate.')
	parser.add_argument('--batch', type=int, default=128, help='Batch size.')
	parser.add_argument('--epoch', type=int, default=10, help='Number of epochs.')
	# parser.add_argument('--ltwo', type=float, default=0.01, help='L2 regularization rate.')
	parser.add_argument('--epochsize', type=int, default=20000, help='How many frames per epoch.')
	parser.add_argument('--validsize', type=int, default=3000, help='How many validation samples.')
	args = parser.parse_args()

	train_features, train_labels = shuffle(train_features, train_labels)
	samples_per_epoch = (args.epochsize//args.batch)*args.batch

	# model = get_comma_model(learning_rate=args.rate)
	model = get_model(learning_rate=args.rate)
	model.fit_generator(
	    generator=generate(train_features, train_labels, args.batch),
	    samples_per_epoch=samples_per_epoch,#args.epochsize,
	    nb_epoch=args.epoch,
	    validation_data=generate(valid_features, valid_labels, args.batch),
	    nb_val_samples=args.validsize
	)

	json = model.to_json()
	model.save_weights('./model.h5')
	with open('./model.json', 'w') as out:
		out.write(json)

	# plot(model, to_file='./model.png')
