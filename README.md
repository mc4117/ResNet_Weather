# ResNet_Weather

This repository contains the code for the following paper

Mariana C. A. Clare, Omar Jamil, Cyril Morcrette, **Using deep learning to produce a computationally efficient probabilistic weather forecast through the prediction of probability density functions**, Quarterly Journal of the Royal Meteorological Society.

This code trains a direct residual convolutional neural network on continuous weather data which has been binned so that a SoftMax layer can be used to predict a probability density function for each output rather than a single value.

The variables predicted are the geopotential at the 500hPa level and the temperature at the 850hPa level.

The repository also contains code to train a stacked neural network to combine outputs from different neural networks.

This code has used the DataGenerator and Padding (for the CNN) from the start notebooks in this [repository](https://github.com/pangeo-data/WeatherBench).

The dataset used is the WeatherBench dataset [1] which is hosted [here](https://mediatum.ub.tum.de/1524895) and more information about how to download the data, as well as the dataset itself, can be found at this [repository](https://github.com/pangeo-data/WeatherBench).
In the folder Data Exploration, this dataset has been visualised and explored.

[1] Stephan Rasp, Peter D. Dueben, Sebastian Scher, Jonathan A. Weyn, Soukayna Mouatadid, and Nils Thuerey, 2020. WeatherBench: A benchmark dataset for data-driven weather forecasting. arXiv: https://arxiv.org/abs/2002.00469

