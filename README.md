# Regression-Linear-Ridge-Lasso-Elastic-
Regression analysis on sample data of House prices


INTRODUCTION
In this task of House Price Prediction using ML, our undertaking is to utilize information from the
dataset provided to make an ML model to foresee house costs. The information incorporates 79
features like Subclass, Zone, Lot area, Street etc. Through this assignment, various preprocessing
techniques like one-hot encoding, scaling are applied. Machine Learning models like Linear
Regressor, Logistic Regressor are applied to preprocessed data for Sales Price Prediction.


METHODOLOGY
This code is written in Python – 3.9.7 using open-source library pandas for loading the raw data
and applying some preprocessing techniques, sklearn for applying other preprocessing techniques
and implementing regression algorithms on preprocessed data to predict sales price.
Objective of this model is to predict the price of house provided all the set of features of
the given house.


Preprocessing
The provided dataset consists of two files train.csv and test.csv each file as 79 various features and
the selling price of the house. The file train.csv consists of 1460 records while the file test.csv
consists of 1460 records. This means this whole dataset is split into train/test in a ratio of 0.5. For
this project we are considering only train.csv for training and testing purpose while test.csv for
evaluation of the model created.


One-Hot Encoding
For categorical variables where no such ordinal relationship exists, the integer encoding is not
enough. In fact, using this encoding and allowing the model to assume a natural ordering between
categories may result in poor performance or unexpected results (predictions halfway between
categories). In this case, a one-hot encoding can be applied to the integer representation. This is
where the integer encoded variable is removed and a new binary variable is added for each unique
integer value. In the “color” variable example, there are 3 categories and therefore 3 binary
variables are needed. A “1” value is placed in the binary variable for the color and “0” values for
the other colors.

Below are the features that contains Categorical data.
Firstly ‘object’ d-type columns are converted to pandas categorical d-type using below code.
Using below Code Categorical Data has been converted into One-Hot Encoded derived features
and featured are stored in variable x and Sales Price is stored in y.

Scaling
Feature scaling is essential for machine learning algorithms that calculate distances between data.
If not scale, the feature with a higher value range starts dominating when calculating distances.
The ML algorithm is sensitive to the “relative scales of features,” which usually happens when it
uses the numeric values of the features rather than say their rank. Normalization is used when we
want to bound our values between two numbers, typically, between [0,1] or [-1,1]. While
Standardization transforms the data to have zero mean and a variance of 1, they make our data
unitless.

Min-Max Scalar
Transform features by scaling each feature to a given range. This estimator scales and translates
each feature individually such that it is in the given range on the training set, e.g., between zero
and one. This Scaler shrinks the data within the range of -1 to 1 if there are negative values. We
can set the range like [0,1] or [0,5] or [-1,1]. This Scaler responds well if the standard deviation
is small and when a distribution is not Gaussian. This Scaler is sensitive to outliers.

Min-Max Scalar has been applied to the dataset as shown in below code.
Records in train dataset has been split into train and test using train_test_split function
sklearn.model_selection library. Below is the code to split the dataset to train/test splits.


MODEL BUILDING
Linear Regression
Below is the Linear Regression implementation of the preprocessed training dataset.
From the RMSE and MAE values, we can say that Linear Regression didn’t perform well on the
dataset.


Regression
Ridge Regressor:
Below is the implementation of Ridge Regressor of the preprocessed dataset.
The RMSE and MAE value of Ridge model is less than Linear Model. Hence we can say that
Ridge performed better than Linear Regression.

Lasso Regressor:
The RMSE and MAE value of Lasso model is less than Ridge Model. Hence, we can say that
Lasso performed better than above two models.

Elastic Net Regressor
Below is the implementation of Elastic Net Regressor on the training data.
From the RMSE and MAE values we can say that Elastic Net didn’t perform well than Ridge
and Lasso models.

Hence, we can say that Lasso model can be used for prediction. Below is the evaluation on the
evaluation dataset.

Evaluation of Test Data
For Evaluation first we need to load the test data into a pandas data frame. Then we need to apply
all the preprocessing steps we did on the training data. Later we need to apply the transform we
already stored during training for standardizing the evaluation data. Then this transformed data
can be passed to the model trained to calculate the sales prediction.

