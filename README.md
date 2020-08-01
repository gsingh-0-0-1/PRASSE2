# PRASSE2

Link to the first (non-machine-learning) PRASSE: https://github.com/gsingh-0-0-1/PRASSE

In short, this repository is dedicated to cutting down the time we spend filtering through processed data from radio telescopes to look for pulsars.

I've employed a pretty basic two-layer (one input, one output) neural network to solve this. As of now, it seems to be working very efficiently and very accurately - it can analyze ~40 plots per second and has yet to classify any of those plots incorrectly.

To clarify - this neural network looks specifically at the phase versus frequency - or phase versus sub-band - plot. The plots that it was trained on and the plots that I have tested the resulting network with are all obtained from the Pulsar Search Collaboratory (PSC), run out of West Virginia University and Green Bank Observatory. Unfortunately, this does end up limiting the format of the data that is inputted into this network - specifically, if your image file names do not contain the string "phasesubband", the network will assume that those files are plots from the PSC, and attempt to crop out the phase-sub-band plot from it. The network also assumes that it will be looking at two full phases in the plot.

However, the accuracy of this network on data that it can analyze is a proof of concept - that neural networks can be extremely efficient and accurate in cutting down the time taken for the analysis of processed radio telescope data.

As of now, I'm working to make this network more flexible, so that it can take in image data not specifically formatted like the PSC's data and still perform at the same accuracy. The end goal would be some type of GUI, where the user could load in their data, specify some basic information about the data, and then let the network analyze and sort the data.

To get started with this code, just ensure that you have python installed, as well as the libraries ```numpy```, ```cv2```, ```PIL```, and ```matplotlib```.```matplotlib``` is not currently needed for the classifier to run, but I have included it as a requirement, as I believe it will be very useful in the future. The rest of the libraries should be installed with python be default.
