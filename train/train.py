# USAGE
# python train.py -d ../data/train/ -ep 200 -batch 32

# import the necessary packages
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from model import model
import cv2
import numpy as np
import argparse
import locale
import os
from datetime import datetime

# construct the argument parser and parse the arguments
def get_argument():

	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", type=str, required=True,
		help="path to input dataset of house images")
	ap.add_argument("-ep", "--epochs", type=int, required=True,
		help="The number loop through all samples")
	ap.add_argument("-batch", "--batch_size", type=int, required=True,
		help="The number of samples at once.")
	args = vars(ap.parse_args())
	return args

def load_dataset(path: str):	
	
	df = []
	images = []

	print("[INFO] loading image attributes...")
	for folder in os.listdir(path):
		if "map" in folder:
			grouth_truth_file = os.path.join(path, "grouth_truth", folder+".txt")
			image_folder = os.path.join(path, folder)
			with open(grouth_truth_file, 'r') as f:
				f.readline()
				for line in f.readlines():
					index, pre_angle, pre_speed, curr_angle, curr_speed = line.split(",")
					try:
						image = cv2.imread(os.path.join(image_folder, str(index)+".png"))
						images.append(image)
						df.append((float(pre_angle), float(pre_speed), float(curr_angle), float(curr_speed)))
					except:
						pass

	return df, images


def preprocess(df, images):
	images = images / 255.0

	# partition the data into training and testing splits using 75% of
	# the data for training and the remaining 25% for testing
	split = train_test_split(df, images, test_size=0.25, random_state=42)
	(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

	# find the largest house price in the training set and use it to
	# scale our house prices to the range [0, 1] (will lead to better
	# training and convergence)
	maxPrice = trainAttrX["price"].max()
	trainY = trainAttrX["price"] / maxPrice
	testY = testAttrX["price"] / maxPrice

	return trainAttrX, trainImagesX, trainY, testAttrX, testImagesX, testY


def main(args):

	# Load dataset
	df, images = load_dataset(path=args['dataset'])
	print("---> The number of images in dataset: {}".format(len(df)))

	print(df[0])
	cv2.imshow("Hiahi", images[0])
	cv2.waitKey(0)

	# Preprocess dataset
	trainAttrX, trainImagesX, trainY, testAttrX, testImagesX, testY = preprocess(df=df, images=images)

	# Create CNN model
	model = models.create_cnn(64, 64, 3, regress=True)
	opt = Adam(lr=1e-3, decay=1e-3 / 200)
	model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

	# Train the model
	print("[INFO] training model...")
	model.fit(x=trainImagesX, y=trainY, 
	    validation_data=(testImagesX, testY),
	    epochs=args['epochs'], batch_size=args['batch_size'])

	# Make predictions on the testing data
	print("[INFO] predicting house prices...")
	preds = model.predict(testImagesX)

	# compute the difference between the *predicted* house prices and the
	# *actual* house prices, then compute the percentage difference and
	# the absolute percentage difference
	diff = preds.flatten() - testY
	percentDiff = (diff / testY) * 100
	absPercentDiff = np.abs(percentDiff)

	# compute the mean and standard deviation of the absolute percentage
	# difference
	mean = np.mean(absPercentDiff)
	std = np.std(absPercentDiff)

	# finally, show some statistics on our model
	locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
	print("[INFO] avg. house price: {}, std house price: {}".format(
		locale.currency(df["price"].mean(), grouping=True),
		locale.currency(df["price"].std(), grouping=True)))
	print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))

	# Save model to disk
	name_model = os.path.join("../saved_models/", datetime.now().strftime("%m-%d-%H") + ".h5")
	model.save(name_model)

if __name__ == "__main__":
	args = get_argument()
	main(args)