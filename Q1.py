### Import useful packages
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns; sns.set()
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

### This section of the code generates the training data for fitting the Gaussian Process Regression model. 
x_end = 2                                                                               ### End point of the training set interval 
train_datapoints = 200                                                                  ### Number of extracted training set data points 
X_n = np.linspace(start = 0, stop = x_end, num = train_datapoints)                        ### Generating the actual data points 

### Define a simple function f(x)
def f(X_n):
    f = np.sin((2 * np.pi) * X_n)                                                         ### f(x) = sin(2πx)
    return (f)

### Store results in variable f_x
f_x = f(X_n)

### Adding noise to the functional evaluations 
sigma_noise = 0.1                                                                       ### Sigma noise is the standard deviation of the normal distribution from which the random numbers are sampled
epsilon_n = np.random.normal(loc = 0, scale = sigma_noise, size = train_datapoints)   ### Adding the noise to each training datapoint 
y_n = f_x + epsilon_n                                                            ### y values of the training set is now the function + error in the training dataset

### Visualise the training data with a plot
fig, ax = plt.subplots(figsize=(12, 5))
sns.scatterplot(x = X_n, y = y_n, label = "training data", ax = ax);                  ### Scatterplot of the training set datapoints with added noise
sns.lineplot(x = X_n, y = f_x, color = "red", label = "f(x)", ax = ax);                   ### Lineplot of the noise-free function
ax.set(title ="Training data for GPR model")
ax.legend(loc = "upper right")
ax.set(xlabel = "x", ylabel = "y")
plt.show()

