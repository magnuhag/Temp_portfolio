import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from NN import NeuralNet
import time

def bias(data, model):
    """
    function for computing bias of data
    """
    return np.mean((data-np.mean(model))**2)
def variance(data, model):
    """
    function for computing variance of data
    """
    return np.mean((model-np.mean(model))**2)


"""
Preparing the data:
"""

#Loading in the data into a pandas dataframe
dataset = pd.read_csv("2019.csv")


#Our target is the Score.
y_a = dataset["Score"]
#Converting y_a into array
y = np.asarray([y_a[i] for i in range(len(y_a))])

#Useless metric
dataset.pop("Overall rank")
#Don't want our target in the design matrix
dataset.pop("Score")
#Don't need this one
dataset.pop("Country or region")
#Creating design matrix
X = dataset.values


"""
Principal component analysis
"""

scaler = StandardScaler()
#Scaling the transpose of X for PCA
scaler.fit(X.T)
X_scaled = scaler.transform(X.T)
#Doing PCA
pca = PCA()
pca.fit(X_scaled.T)
pca_data = pca.transform(X_scaled.T)
#Rounding off the percentages
per_val = np.round(pca.explained_variance_ratio_*100, decimals = 1)
#Labels of PC for plotting
labels = dataset.keys()
labels2 = np.linspace(1,len(labels), len(labels))

#Making a bar plot of the contribution of each principal component
plt.bar(x=range(1,len(per_val)+1), height = per_val, tick_label = labels2)
plt.title("Principal Component Analysis of dataset")
plt.ylabel("Percentage of explained values")
plt.xlabel("Principal component")
#plt.show()

X_full = dataset.values

#Deleting PC6 from dataset to create reduced dataset
dataset.pop("Perceptions of corruption")

X_reduced = dataset.values

#For plotting GDP per captia against happiness score
gdp = dataset["GDP per capita"]


plt.plot(gdp, y, "o")
plt.xlabel("GPD per capita")
plt.ylabel("Happiness Score")
plt.title("Happiness as a function of wealth")
#plt.show()

"""
Model fitting: Linear regression on full dataset
"""

X = X_full

X_train, X_test, y_train, y_test = train_test_split(X, y)

#Simple linear regression
reg = LinearRegression(fit_intercept = False).fit(X_train, y_train)
pred = reg.predict(X_test)


print("train MSE full dataset = %g" %np.mean((y_train-X_train@reg.coef_)**2))
print("test MSE full dataset = %g" %np.mean((y_test-pred)**2))

"""
Model fitting: Linear regression on reduced dataset
"""

X = X_reduced

X_train, X_test, y_train, y_test = train_test_split(X, y)


reg = LinearRegression(fit_intercept = False).fit(X_train, y_train)
pred = reg.predict(X_test)

#Comparing regression with and without PC6
print("train MSE reduced dataset = %g" %np.mean((y_train-X_train@reg.coef_)**2))
print("test MSE reduced dataset = %g" %np.mean((y_test-pred)**2))

"""
Model fitting: RidgeCV and LassoCV, on full dataset
"""

X = X_full
X_train, X_test, y_train, y_test = train_test_split(X, y)
#Regularization parameters for Ridge and Lasso regression
alphass = np.logspace(-5, 1, 100)
epss = np.logspace(-7, -1, 40)


#Ridge regression with cross validation
reg_ridge = RidgeCV(alphas = alphass, fit_intercept = False).fit(X_train, y_train)
pred_ridge = reg_ridge.predict(X_test)

#Finding mse of ridge
print("train MSE Ridge full dataset = %g" %np.mean((y_train-X_train@reg_ridge.coef_)**2))
print("test MSE Ridge full dataset= %g" %np.mean((y_test-pred_ridge)**2))

#for plotting Lasso MSE as a function of reg.parameter
mse_train = np.zeros(len(epss))
mse_test = mse_train.copy()
for i in range(len(epss)):
    reg_lasso = LassoCV(eps = epss[i], n_alphas = 500, fit_intercept = False).fit(X_train, y_train)
    pred_lasso = reg_lasso.predict(X_test)
    mse_train[i] = np.mean((y_train-X_train@reg_lasso.coef_)**2)
    mse_test[i] = np.mean((y_test-pred_lasso)**2)


plt.plot(epss, mse_train, label = "test")
plt.plot(epss, mse_test, label = "train")
plt.title("MSE of LassoCV regression as function of eps, full dataset")
plt.xlabel("eps")
plt.ylabel("MSE")
plt.legend()
plt.show()


"""
Model fitting: RidgeCV and LassoCV, on reduced dataset
"""

X = X_reduced
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Regularization parameters for Ridge and Lasso regression
alphass = np.logspace(-5, 1, 100)
epss = np.logspace(-7, -1, 40)


#Ridge regression with cross validation
reg_ridge = RidgeCV(alphas = alphass, fit_intercept = False).fit(X_train, y_train)
pred_ridge = reg_ridge.predict(X_test)

#Finding mse of ridge
print("train MSE Ridge reduced dataset = %g" %np.mean((y_train-X_train@reg_ridge.coef_)**2))
print("test MSE Ridge reduced dataset = %g" %np.mean((y_test-pred_ridge)**2))

#for plotting Lasso MSE as a function of reg.parameter
mse_train = np.zeros(len(epss))
mse_test = mse_train.copy()
for i in range(len(epss)):
    reg_lasso = LassoCV(eps = epss[i], n_alphas = 500, fit_intercept = False).fit(X_train, y_train)
    pred_lasso = reg_lasso.predict(X_test)
    mse_train[i] = np.mean((y_train-X_train@reg_lasso.coef_)**2)
    mse_test[i] = np.mean((y_test-pred_lasso)**2)


plt.plot(epss, mse_train, label = "test")
plt.plot(epss, mse_test, label = "train")
plt.title("MSE of LassoCV regression as function of eps, reduced dataset")
plt.xlabel("eps")
plt.ylabel("MSE")
plt.legend()
plt.show()

"""
#FFNN with full dataset
"""

# We first try to use the NN with PC6 in the dataset
X = X_full
# Suitable number of iterations
num_iters = 400
eta = 10e-4
#activation function. Choose between "Sigmoid", "RELU", and "Leaky_RELU"
act = "RELU"
#g = "reg" means regression. g="clas" means classification
g = "reg"
#regularization parameter
lam = 1
#Whether to use mini-batches. mini = [True/ False, n_batches, n_epochs]
#If no mini-batch: mini = [False]
mini = [True, 60, 100]


#Number of input nodes. Must match number of features in X
h = int(len(X[0]))

#Architecture the was found (by trial and error) to best work in this with this data
#layer_sizes = [input_layer, hidden_layer_1, ..., hidden_layer_n, outputlayer]
layer_sizes = [h, h, 1]
#List of possible activation functions
funcs = ["Sigmoid", "RELU", "Leaky_RELU"]
#For timing the execution time of the NN
start_time = time.time()
for func in funcs:

    net = NeuralNet(X, y, layer_sizes, num_iters, eta, func, g, lam, mini, scale = True)
    params = net.model()
    acc_train, acc_test = net.compute_accuracy(params)
    print("execution time with " + func + " function is " + str((time.time() - start_time))+ ", full dataset")
    print("training MSE full dataset = " +str(acc_train))
    print("test MSE full dataset = " +str(acc_test))

"""
FFNN with reduced dataset
"""

#Dataset without PC6
X = X_reduced



h = int(len(X[0]))

layer_sizes = [h, h, 1]

for func in funcs:
    start_time = time.time()
    net = NeuralNet(X, y, layer_sizes, num_iters, eta, func, g, lam, mini, scale = True)
    params = net.model()
    acc_train, acc_test = net.compute_accuracy(params)
    print("execution time with " + func + " function is " + str((time.time() - start_time))+ ", reduced dataset")
    print("training MSE reduced dataset = " +str(acc_train))
    print("test MSE reduced dataset = " +str(acc_test))


"""
Regression tree with reduced dataset
"""


X = X_reduced
X_train, X_test, y_train, y_test = train_test_split(X, y)

#max possible depth of tree
n = 14
mse_test = np.zeros(n)
Bias = mse_test.copy()
Variance = Bias.copy()
depth = np.linspace(1,n, n)
depth = np.array([int(i) for i in range(1, len(depth)+1)])
start_time = time.time()
for i in depth:
    regressor = DecisionTreeRegressor(max_depth = i).fit(X_train, y_train)
    pred = regressor.predict(X_test)
    Bias[i-1] = bias(y_test, pred)
    Variance[i-1] = variance(y_test, pred)
    mse_test[i-1] = np.mean((y_test-pred)**2)

print("execution time with reduced dataset = "  + str((time.time() - start_time)))
plt.plot(depth, mse_test, label = "MSE")
plt.plot(depth, Bias, label = "Bias")
plt.plot(depth, Variance, label = "Variance")
plt.legend()
plt.title("Regression tree: : Bias Variance as a function of depth, reduced dataset")
plt.xlabel("Depth")
plt.ylabel("MSE")
plt.show()


"""
Regression tree with full dataset
"""

X = X_full
X_train, X_test, y_train, y_test = train_test_split(X, y)

n = 14
mse_test = np.zeros(n)
Bias = mse_test.copy()
Variance = Bias.copy()
depth = np.linspace(1,n, n)
depth = np.array([int(i) for i in range(1, len(depth)+1)])
start_time = time.time()
for i in depth:
    regressor = DecisionTreeRegressor(max_depth = i).fit(X_train, y_train)
    pred = regressor.predict(X_test)
    Bias[i-1] = bias(y_test, pred)
    Variance[i-1] = variance(y_test, pred)
    mse_test[i-1] = np.mean((y_test-pred)**2)

print("execution time with reduced dataset = "  + str((time.time() - start_time)))
plt.plot(depth, mse_test, label = "MSE")
plt.plot(depth, Bias, label = "Bias")
plt.plot(depth, Variance, label = "Variance")
plt.legend()
plt.title("Regression tree: : Bias Variance as a function of depth, full dataset")
plt.xlabel("Depth")
plt.ylabel("MSE")
plt.show()



"""
Random forest reduced dataset
"""

X = X_reduced

n = 14

X_train, X_test, y_train, y_test = train_test_split(X, y)

depth = np.linspace(1,n, n)
depth = np.array([int(i) for i in range(1, len(depth)+1)])

mse_test = np.zeros(n)
Bias = mse_test.copy()
Variance = Bias.copy()
start_time = time.time()
for i in depth:
    reg = RandomForestRegressor(max_depth = i).fit(X_train, y_train)
    pred = reg.predict(X_test)
    mse_test[i-1] = np.mean((y_test-pred)**2)
    Bias[i-1] = bias(y_test, pred)
    Variance[i-1] = variance(y_test, pred)
print("execution time with reduced dataset = "  + str((time.time() - start_time)))
plt.plot(depth, mse_test, label = "MSE")
plt.plot(depth, Bias, label = "Bias")
plt.plot(depth, Variance, label = "Variance")
plt.legend()
plt.title("Random forest: Bias Variance as a function of depth, reduced dataset")
plt.xlabel("Depth")
plt.ylabel("MSE")
plt.show()

"""
Random forest full dataset
"""

X = X_full

n = 14

X_train, X_test, y_train, y_test = train_test_split(X, y)

depth = np.linspace(1,n, n)
depth = np.array([int(i) for i in range(1, len(depth)+1)])

mse_test = np.zeros(n)
Bias = mse_test.copy()
Variance = Bias.copy()
start_time = time.time()

for i in depth:
    reg = RandomForestRegressor(max_depth = i).fit(X_train, y_train)
    pred = reg.predict(X_test)
    mse_test[i-1] = np.mean((y_test-pred)**2)
    Bias[i-1] = bias(y_test, pred)
    Variance[i-1] = variance(y_test, pred)

print("execution time with full dataset = "  + str((time.time() - start_time)))
plt.plot(depth, Bias, label = "Bias")
plt.plot(depth, Variance, label = "Variance")
plt.plot(depth, mse_test, label = "MSE")
plt.legend()
plt.title("Random forest: Bias Variance as a function of depth, full dataset")
plt.xlabel("Depth")
plt.ylabel("Magnitude")
plt.show()


"""
Bagging (Trees)
"""

X = X_full
X_train, X_test, y_train, y_test = train_test_split(X, y)

n = 50
Bias = np.zeros(n)
Variance = np.zeros(n)
mse_test = Bias.copy()

ns = np.linspace(1,n,n)
for i in range(50):
    reg = BaggingRegressor(n_estimators = int(i)+1).fit(X_train, y_train)
    pred = reg.predict(X_test)
    mse_test[i] = np.mean((y_test-pred)**2)
    Bias[i] = bias(y_test, pred)
    Variance[i] = variance(y_test, pred)
plt.plot(ns, Bias, label = "bias")
plt.plot(ns, Variance, label = "variance")
plt.plot(ns, mse_test, label = "MSE test")
plt.title("Bagging: Bias Variance Trade-off with training MSE")
plt.xlabel("Number of trees")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

"""
Gradient Boosting
"""


X = X_full
X_train, X_test, y_train, y_test = train_test_split(X, y)

n = 50

Bias = np.zeros(n)
Variance = np.zeros(n)
mse_test = Bias.copy()
#max_depth = 3
ns = np.linspace(1,n,n)
for i in range(50):
    reg = GradientBoostingRegressor(n_estimators = int(i)+1).fit(X_train, y_train)
    pred = reg.predict(X_test)
    mse_test[i] = np.mean((y_test-pred)**2)
    Bias[i] = bias(y_test, pred)
    Variance[i] = variance(y_test, pred)
plt.plot(ns, Bias, label = "bias")
plt.plot(ns, Variance, label = "variance")
plt.plot(ns, mse_test, label = "MSE test")
plt.title("Gradient Boost: Bias Variance Trade-off with training MSE")
plt.xlabel("Number of trees")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

"""
AdaBoost
"""

X = X_full
X_train, X_test, y_train, y_test = train_test_split(X, y)

n = 50

Bias = np.zeros(n)
Variance = np.zeros(n)
mse_test = Bias.copy()
#max_depth = 3
ns = np.linspace(1,n,n)
for i in range(50):
    reg = AdaBoostRegressor(n_estimators = int(i)+1, learning_rate = 0.1).fit(X_train, y_train)
    pred = reg.predict(X_test)
    mse_test[i] = np.mean((y_test-pred)**2)
    Bias[i] = bias(y_test, pred)
    Variance[i] = variance(y_test, pred)
plt.plot(ns, Bias, label = "bias")
plt.plot(ns, Variance, label = "variance")
plt.plot(ns, mse_test, label = "MSE test")
plt.title("AdaBoost: Bias Variance Trade-off with training MSE")
plt.xlabel("Number of trees")
plt.ylabel("Magnitude")
plt.legend()
plt.show()
