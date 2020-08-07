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

print("Done importing.")
print()

barWidth = 50
def updateProgressBar(value):
    line = '\r%s%%[%s]' % ( str(value).rjust(3), '-' * int ((float(value)/100) * barWidth))  
    print (line, end='')
    sys.stdout.flush()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

print("Please select a model to load (enter the corresponding number - 1, 2...): ")
print("[1] arc_base_1_kernel_3_3")
print("[2] arc_base_1_kernel_3_1")
model_to_load = input(": ")

if model_to_load == "1":
	model_to_load = "arc_base_1_kernel_3_3"
if model_to_load == "2":
	model_to_load = "arc_base_1_kernel_3_1"

model = tf.keras.models.load_model(model_to_load)

print("Please enter a path containing the plots to be classified:")

print("Relative paths should not begin with a '/', but absolute paths should.")

test_data_directory = input("All paths should end with a '/': ")

items = 0

l = []
names = []

print("Loading data...")

num_plots = len(os.listdir(test_data_directory))

load_start_time = time.time()

for fname in os.listdir(test_data_directory):
	if fname[0] == '.' or fname == 'temp.png' or 'single' in fname:
		continue
	try:
		img = Image.open(test_data_directory+fname)
		img = np.array(img)
		if len(np.shape(img)) < 3:
			print ("Input Shape Error // Non-fatal // Skipping to next input")
			continue
	except OSError:
		print("OSError // Non-fatal // Skipping to next input")
		continue


	img = Image.open(test_data_directory+fname)
	phasesubband = np.array(img)[170:355, 320:470][:, :, :3]

	#if np.shape(phasesubband)[2] > 3:
	#	phasesubband = phasesubband[:, :, :3]

	l += [phasesubband]
	names += [fname]

	items += 1

	if items >= 1000:
		break

	updateProgressBar( round((items/num_plots) * 100, 1) )

load_end_time = time.time()

print()

print("Data loaded. Classifying...")

l = np.array(l)

start_time = time.time()

predicted = model.predict_classes(l)

end_time = time.time()

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
	if len(displayname) > 30:
		displayname = fname[:10] + "..." + fname[-15:]
	while len(displayname) < 30:
		displayname = displayname + "-"

	print(displayname + "\t" + str(output[item, 1]))


print("Pulsars: " + str(counts["Pulsar"]))
print("Not Pulsars: " + str(counts["Not Pulsar"]))
print("Classification time: " + "\t" + str(1000 * (end_time - start_time)) + "ms")
print("Items: " + str(items))
print("Avg Load Rate: " + str(round(items / (load_end_time - load_start_time), 3)) + " plots / s")
print("Avg Classification Rate: " + str(round(items / (end_time - start_time), 3)) + " plots / s")


