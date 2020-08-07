import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

args = np.array([0.20900506288154352, 
				0.99998432930365, 
				-1.4651947033133499, 
				0.9999999999229094, 
				0.23339975450270492])

def f2(l):
	temp = 0
	l = [1] + l
	for i in range(len(l)):
		temp += - f2args[i] * l[i]
	return (1 / (1 + np.e**(temp) ) )

def f(x, z):
    return (1 / (1 + np.e**(-args[0] - args[1]*x - args[2]*z - args[3]*x*x - args[4]*z*z ) ) )

def getfeatures(phasesubband):
	ma = np.amax(phasesubband)
	med = np.median(phasesubband)
	std = np.std(phasesubband)
	above = np.where(phasesubband > med+10)

	xlist = np.sum(np.sum(255-phasesubband, axis=2), axis=0)
	xlist = np.roll(xlist, -np.argmax(xlist))
	x1 = np.log(np.amax(xlist) / np.median(xlist)) #kind of SNR
	x2 = np.amax(xlist)

	#some basic feature scaling
	while x1 > 10 or x2 > 10:
		x1 = x1/10
		x2 = x2/10

	return x1, x2

def neurnetfeatures(phasesubband):
	#phasesubband = cv2.resize(phasesubband, (37, 30))
	mi = np.argmin(np.sum(np.sum(phasesubband, axis=2), axis=0))
	new = np.roll(phasesubband, -mi, axis=1)
	new = np.sum(new, axis=2) / 3
	new = new.flatten()
	new = new/255
	new = new - 0.5
	new = np.concatenate(([1], new))
	return new

def neurnetfeatures2(phasesubband):
	phasesubband = cv2.resize(phasesubband, (37, 30))
	mi = np.argmin(np.sum(np.sum(phasesubband, axis=2), axis=0))
	new = np.roll(phasesubband, -mi, axis=1)
	new = np.sum(new, axis=2) / 3
	new = new.flatten()
	new = new/255
	new = new - 0.5
	new = np.concatenate(([1], new))
	return new
