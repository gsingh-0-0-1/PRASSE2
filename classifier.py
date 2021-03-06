#* Copyright (C) Gurmehar Singh 2020 - All Rights Reserved
#* Unauthorized copying or distribution of this file, via any medium is strictly prohibited
#* Proprietary and confidential
#* Written by Gurmehar Singh <gurmehar@gmail.com>, October 2020
#*/


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image
import sys
from getfeatures import getfeatures
from getfeatures import neurnetfeatures
from getfeatures import neurnetfeatures2
import time
import shutil
import cv2

print("Initializing classifier...")

def plotIsBad(fname):
    bad = False
    if fname[0] =='.':
        bad = True
    if fname == 'temp.png':
        bad = True
    if 'single' in fname:
        bad = True
    if os.path.exists('training/notpulsar'):
        if fname in os.listdir('training/notpulsar/'):
            bad = True
    if os.path.exists('training/pulsar'):
        if fname in os.listdir('training/pulsar/'):
            bad = True
    if fname == "classified_pulsar" or fname == "classified_notpulsar":
        bad = True
    return bad

print("Plot-checker loaded...")

thetas = np.loadtxt("neurnet.txt")

print("Parameters/weights loaded...")

print("Please enter a path containing the plots to be classified:")

print("Relative paths should not begin with a '/', but absolute paths should.")

startdir = input("All paths should end with a '/': ")

print()
print("Two directories - classified_pulsar/ and classified_notpulsar/ - ", end='')
print("will be created in the specified path, and the plots will be sorted into those.")
print()

if not os.path.exists(startdir+'classified_pulsar/'):
    os.mkdir(startdir+'classified_pulsar/')
if not os.path.exists(startdir+'classified_notpulsar/'):
    os.mkdir(startdir+'classified_notpulsar/')

p = 0
notp = 0

for fname in os.listdir(startdir):
    if plotIsBad(fname):
        continue
    try:
        img = Image.open(startdir+fname)
        img = np.array(img)
        if len(np.shape(img)) < 3:
            print(fname, np.shape(img))
            print ("error")
            break
    except OSError:
        print(fname)
        print('error')
        break

    phasesubband = img[170:355, 320:470]

    #phasesubband = cv2.resize(phasesubband, dsize=(150, 185))

    f = neurnetfeatures(phasesubband)
    final = np.dot(f, thetas)
    val = 1 / (1 + np.e ** final)
    displayname = fname
    if len(displayname) > 30:
        displayname = fname[:10] + "..." + fname[-15:]
    while len(displayname) < 30:
        displayname = displayname + "-"

    if val >= 0.5:
        pred = "Pulsar"
        p += 1
        shutil.move(startdir+fname, startdir+"classified_pulsar/"+fname)
    else:
        pred = "Not Pulsar"
        notp += 1
        shutil.move(startdir+fname, startdir+"classified_notpulsar/"+fname)

    name_data = "Predicted: "+str('%f' % val)[:10]+"\t ->\t "+str(round(val))+" -> "+pred
    print(displayname+'\t'+name_data)

print("Classified Pulsars: "+str(p))
print("Classified Not-pulsars: "+str(notp))

