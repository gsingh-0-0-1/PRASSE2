# PRASSE2

Link to the first (non-machine-learning) PRASSE: https://github.com/gsingh-0-0-1/PRASSE

In short, this repository is dedicated to cutting down the time we spend filtering through processed data from radio telescopes to look for pulsars.

For the first part of this project, I've employed a pretty basic two-layer (one input, one output) neural network to solve this. As of now, it seems to be working very efficiently - though the accuracy can waver from time to time. Neural networks, by nature, pick up on "features" that we can't really predict as well as we'd like, so as I continue to improve, re-train, and re-design the architecture of the neural network, the accuracy should improve.

To clarify - this neural network looks specifically at the phase versus frequency - or phase versus sub-band - plot. The plots that it was trained on and the plots that I have tested the resulting network with are all obtained from the Pulsar Search Collaboratory (PSC), run out of West Virginia University and Green Bank Observatory. Unfortunately, this does end up limiting the format of the data that is inputted into this network - specifically, if your image file names do not contain the string "phasesubband", the network will assume that those files are plots from the PSC, and attempt to crop out the phase-sub-band plot from it. The network also assumes that it will be looking at two full phases in the plot.

However, the accuracy of this network on data that it can analyze is a proof of concept - that neural networks can be extremely efficient and accurate in cutting down the time taken for the analysis of processed radio telescope data.

As of now, I'm working to make this network more flexible, so that it can take in image data not specifically formatted like the PSC's data and still perform at the same accuracy. The end goal would be some type of GUI, where the user could load in their data, specify some basic information about the data, and then let the network analyze and sort the data.

The second part of this project revolves around more complex, multilayer networks. Given that constructing the one-layer network by hand was difficult enough by itself, I decided to go the TensorFlow and Keras route for constructing more complex, powerful networks. The variety of layers, functions, and operations that you can manipulate really create a lot of options to change how effectively you can analyze data using any model you design.

I've designed two models so far - the main difference is in the kernel size (or shape, rather). The model arc_base_1_kernel_3_3 uses, not surprisingly, a kernel size of (3, 3) in its convolutional layers. This is relatively standard, and the validation accuracy wasn't half-bad - around 95%. But, given that what we are looking for here are effectively vertical lines, we can modify the kernel size to fit to the directional nature of the data. So, the second model - arc_base_1_kernel_3_1 - uses a kernel size of (3, 1), which I initially thought would help with the validation accuracy and overall detection - and given that the validation accuracy rose to nearly 98%, it seems to have worked. 

To run either of these models, just run ```multilayer_classifier.py```, and that will walk you through selecting a model and inputting test data.

The essence here is that, now, the focus is on finding the ideal architecture for a model. I'll continue to play around with the architecture and update this repo with whatever I find.

In order to run the models, make sure that you have python installed, as well as the libraries ```tensorflow```, ```keras```, ```numpy```, ```cv2```, ```PIL```, and ```matplotlib```.```matplotlib``` is not currently needed for the classifier to run, but I have included it as a requirement, as I believe it will be very useful in the future. The rest of the libraries should be installed with python be default.
