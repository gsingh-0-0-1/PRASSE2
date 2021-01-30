#* Copyright (C) Gurmehar Singh 2020 - All Rights Reserved
#* Unauthorized copying or distribution of this file, via any medium is strictly prohibited
#* Proprietary and confidential
#* Written by Gurmehar Singh <gurmehar@gmail.com>, October 2020
#*/

print("Importing libraries...")


import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np
import os
import PIL
from PIL import Image
import logging
import sys
import time
import pydot
import pydotplus
from pydotplus import graphviz
import multiprocessing
import cv2
import glob

print("Done importing.")
print()

barWidth = 50
def updateProgressBar(value):
    line = '\r%s%%[%s]' % ( str(value).rjust(3), '-' * int ((float(value)/100) * barWidth))  
    print (line, end='')
    sys.stdout.flush()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

models = ["arc_base_1_kernel_3_3_padding_same",
		"arc_base_1_kernel_3_1_padding_same",
		"arc_base_1_kernel_3_3_padding_none",
		"arc_base_1_kernel_3_1_padding_none",
		"model_dev"]

print("Please select a model to load (enter the corresponding number - 1, 2...): ")
for model in range(len(models)):
	print("[" + str(model + 1) + "]" + " " + str(models[model]))

model_to_load = input(": ")

model_to_load = models[int(model_to_load)-1]

model_to_load = 'models/' + model_to_load

model = tf.keras.models.load_model(model_to_load)

print("Please enter a path containing the plots to be classified:")

print("Relative paths should not begin with a '/', but absolute paths should.")

test_data_directory = input("All paths should end with a '/': ")

items = 0

l = []
names = []

print("Loading data...")

num_plots = len([i for i in os.listdir(test_data_directory) if not i.startswith(".")])

print()
print("Please specify a limit for the number of plots to load. To set the limit to infinity, press enter.")
max_plots = input(": ")

try:
	max_plots = int(max_plots)
except ValueError:
	max_plots = num_plots

print(max_plots)

batches = int(max_plots / 1000) + 1

total_load_time = 0
total_classify_time = 0

listed_plots = [i for i in os.listdir(test_data_directory) if not i.startswith(".")]

for batch in range(batches):
	l = []
	names = []
	thisitems = 0

	if batch == batches - 1:
		thismax = max_plots % 1000
	else:
		thismax = 1000

	print("------------------------------")
	print("Starting batch " + str(batch + 1) + " / " + str(batches))
	print("------------------------------")

	load_start_time = time.time()

	for plot in range(batch * 1000, batch * 1000 + thismax):
		fname = listed_plots[plot]
		if fname[0] == '.' or fname == 'temp.png' or 'single' in fname:
			continue
		img = cv2.imread(test_data_directory+fname)[170:355, 320:470]
		if len(np.shape(img)) < 3:
			print ("Input Shape Error // Non-fatal // Skipping to next input")
			continue

		l += [img]
		names += [fname]

		thisitems += 1

		updateProgressBar( round((thisitems/thismax) * 100, 1) )

		del(img)

	load_end_time = time.time()

	items += thisitems

	total_load_time += (load_end_time - load_start_time)

	print()

	print("Data loaded. Classifying...")

	l = np.array(l)

	start_time = time.time()

	predicted = model.predict_classes(l)

	end_time = time.time()

	total_classify_time += (end_time - start_time)

	print("Done classifying. Cleaning up output...")

	names = np.array(names)

	output = np.column_stack((names, predicted))

	output[np.where(output == '0')] = "Not Pulsar"
	output[np.where(output == '1')] = "Pulsar"

	unique, counts = np.unique(output, return_counts=True)

	counts = dict(zip(unique, counts))

	if "Pulsar" not in counts.keys():
		counts["Pulsar"] = 0

	if "Not Pulsar" not in counts.keys():
		counts["Not Pulsar"] = 0
	 
	for item in range(len(output)):
		fname = output[item, 0]
		displayname = fname
		if len(displayname) > 100:
			displayname = fname[:50] + "..." + fname[-40:]
		while len(displayname) < 100:
			displayname = displayname + "-"

		print(displayname + "\t" + str(output[item, 1]))


	print("Pulsars: " + str(counts["Pulsar"]))
	print("Not Pulsars: " + str(counts["Not Pulsar"]))
	print("Classification time: " + "\t" + str(1000 * (end_time - start_time)) + "ms")
	print("Items: " + str(thisitems))
	print("Load time: " + "\t" + str(1000 * (load_end_time - load_start_time)) + "ms")
	print("Avg Load Rate: " + str(round(thisitems / (load_end_time - load_start_time), 3)) + " plots / s")
	print("Avg Classification Rate: " + str(round(thisitems / (end_time - start_time), 3)) + " plots / s")

	l = None
	names = None
	output = None
	counts = None
	unique = None

print("------------------------------------------------------------")
print("Finished " + str(batches) + " batches")
print("Overall Statistics:")
print("Classification time: " + "\t" + str(1000 * (total_classify_time)) + "ms")
print("Items: " + str(items))
print("Load time: " + "\t" + str(1000 * (total_load_time)) + "ms")
print("Avg Load Rate: " + str(round(items / (total_load_time), 3)) + " plots / s")
print("Avg Classification Rate: " + str(round(items / (total_classify_time), 3)) + " plots / s")


