# MLTS_Project

MLTS_Project is a data-driven project aimed at predicting electric consumption for charging stations using an open-source dataset from [Boulder, Colorado's open data portal](https://open-data.bouldercolorado.gov/). The dataset contains detailed records of electric vehicle charging transactions at various city-owned charging stations in Boulder. This project explores and compares several machine learning and deep learning techniques, including Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), Bidirectional Long Short-Term Memory (BiLSTM), Linear Regression, Random Forest, and Support Vector Regression (SVR) for the prediction of power consumption based on one year of real electric vehicle load data.

## Project Structure

The repository is organized as follows:

* `main.py`: This file loads the dataset and configures model parameters.
* `utils/utils.py`: Contains utility functions for handling outliers, splitting the dataset into training and testing sets, and calculating evaluation metrics.
* `models/`: This directory contains the implementations of the three deep learning models used in the project.
* `optimization/optimization.py`: Used for training and testing the models.

Feel free to explore the code and experiment with different models for electric consumption prediction. 
