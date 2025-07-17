### Import useful packages
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

### Include a random seed for reproducibility 
np.random.seed(13)

### ==============================================================================================================
### Generate Training Data (Flowchart: "Reference Data")
### ==============================================================================================================

### Code to generate the x locations in the training data 
x_end = 3                                                                   ### End point of the training set interval 
n_train = 200                                                               ### Number of extracted training set datapoints 
x_n = np.linspace(start = 0, stop = x_end, num = n_train)                   ### Generating the training set datapoints

### Define a simple function f(x) and store the results 
def f(x_n):
    f = np.sin((2 * np.pi) * x_n)                                           ### f(x) = sin(2πx)
    return (f) 
f_x = f(x_n)                                                                ### Store the results in variable f_x

### Adding noise to the functional evaluations 
sigma_noise = 0.1                                                           ### Sigma noise is the standard deviation of the normal distribution from which the random numbers are sampled
epsilon_n = np.random.normal(loc = 0, scale = sigma_noise, size = n_train)  ### Adding the noise to each training set datapoint 
y_n = f_x + epsilon_n                                                       ### y values of the training set is now defined as the original function + error in the training dataset

### Optional: Visualise the training data with a plot
plt.figure(figsize = (10, 5))
plt.scatter(x = x_n, y = y_n, label = "training data")                      ### Scatterplot of the training set datapoints with added noise
plt.plot(x_n, f_x, color = "red", label = "f(x)")                           ### Lineplot of the noise-free function
plt.title("Training data for GPR model")
plt.legend(loc = "upper right")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

### ==============================================================================================================
### Build the kernel (Flowchart: "Kernel Construction")
### ==============================================================================================================

### Build a kernel for Gaussian Process Regression
kernel = RBF(length_scale = 1.0)

gpr = GaussianProcessRegressor(
    kernel = kernel,
    alpha = sigma_noise ** 2,                                               ### Regularisation matches the noise variance
    optimizer = None                                                        ### No optimisation of the hyperparameters for simplicity
)

### ==============================================================================================================
### Model Training (Flowchart: "Model Training")
### ==============================================================================================================

### Fit the Gaussian Process Regression model to the training data 
gpr.fit(x_n.reshape(-1, 1), y_n)

### ==============================================================================================================
### Prediction (Flowchart: "Prediction")
### ==============================================================================================================

### Define test locations 
n_test = n_train + 200
x_test = np.linspace(start = 0, stop = (x_end + 1), num = n_test)           ### Letting the test interval be bigger than the training interval helps us determine the validity of the model outside of the training data 

### Predict the mean and variance at new test locations 
y_mean, y_std = gpr.predict(x_test.reshape(-1, 1), return_std = True)     

### Optional: Visualise the predicted values 
plt.figure(figsize = (10, 5))
plt.plot(x_n, f_x, 'r-', label = "True function $f(x)$")
plt.scatter(x_n, y_n, color = "k", s = 20, label = "Training Data")
plt.plot(x_test, y_mean, 'g-', label = "GPR mean prediction")
plt.fill_between(x_test, y_mean - 2 * y_std, y_mean + 2 * y_std, color = "green", alpha = 0.2, label = "95% confidence interval")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gaussian Process Regression (Simple 1D Example)")
plt.legend()
plt.show()