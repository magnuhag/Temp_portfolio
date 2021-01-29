import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class NeuralNet:
    def __init__(self, X_data, Y_data, layer_sizes, num_iters, eta, act, g, lam, mini, scale):



        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_data, Y_data, test_size = 0.2)

        self.layer_sizes = layer_sizes
        self.num_iters = num_iters
        self.eta = eta
        if act == "Sigmoid":
            self.act = lambda x: 1/(1+np.exp(-x))
            self.d_act = lambda x: np.exp(-x)/(1+np.exp(-x))**2
        elif act == "RELU":
            self.act = lambda x: np.maximum(x, 0)
            def d_act(x):
                x[x<=0] = 0
                x[x>0] = 1
                return x
            self.d_act = d_act
        elif act == "Leaky_RELU":
            self.act = lambda x: np.maximum(x, 0.01 * x)
            def d_act(x):
                alpha = 0.01
                dx = np.ones_like(x)
                dx[x < 0] = alpha
                return dx
            self.d_act = d_act


        if scale == True:
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            self.X_train = scaler.transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        self.g = g
        self.lam = lam
        self.mini = mini

    def initialize_params(self):

        params = {}
        #initializing the weights and biases with random values
        #Could be nice to be able to specify intial weights and biases as function arguments,
        #but that might be superfluous
        for i in range(1, len(self.layer_sizes)):
            params["weight" + str(i)] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1])*0.01
            params["bias" + str(i)] = np.random.randn(self.layer_sizes[i],1)*0.01
        return params

    def feed_forward(self, params, X):

        layers = len(params)//2
        values = {}

        for i in range(1, layers+1):
            #feeding data to the input layer
            if i==1:
                #input to node
                values["z" + str(i)] = np.dot(params["weight" + str(i)], X.T) + params["bias" + str(i)]
                #activating node
                values["a" + str(i)] = self.act(values["z" + str(i)])

            #feeding forward to next layer
            else:
                values["z" + str(i)] = np.dot(params["weight" + str(i)], values["a" + str(i-1)]) + params["bias" + str(i)]
                #In the regression case: output values are equal to the input in the output layer
                if i==layers and self.g == "reg":
                    values["a"+ str(i)] = values["z" + str(i)]
                #In the classification case: output values are probabilities of correct prediction
                elif i==layers and self.g == "clas":
                    a = self.act(values["z" + str(i)])
                    exp_term = np.exp(a)
                    probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
                    values["a" + str(i)] = probabilities
                #The activation is equal in both regression and classification, but
                #one should test the different ones for the best case use
                else:
                    values["a" + str(i)] = self.act(values["z" + str(i)])
        return values

    def backprop(self, params, values):
        layers = len(params)//2
        m = len(self.Y_train)
        grads = {}

        #In the regression case:
        if self.g == "reg":
            #Starting at the output layer, going back
            for i in range(layers,0,-1):
                #Finding MSE of output layer
                #Might add different cost function options
                if i==layers:
                    dA = 1/m * np.sum((values["a" + str(i)] - self.Y_train)**2, axis=0, keepdims = True)
                    dZ = dA
                #Propage error backward
                else:
                    dA = np.dot(params["weight" + str(i+1)].T, dZ)
                    dZ = np.multiply(dA, self.d_act(values["a" + str(i)]))
                #If at input layer
                if i==1:
                    grads["weight" + str(i)] = 1/m*np.dot(dZ, self.X_train)
                    #Regularization term. If lam = 0: no regularization
                    grads["weight" + str(i)] += self.lam * params["weight" + str(i)]
                    grads["bias" + str(i)] = 1/m*np.sum(dZ, axis=1, keepdims=True)
                #Else at hidden hidden layer
                else:
                    grads["weight" + str(i)] = 1/m*np.dot(dZ,values["a" + str(i-1)].T)
                    #Regularization term
                    grads["weight" + str(i)] += self.lam * params["weight" + str(i)]
                    grads["bias" + str(i)] = 1/m*np.sum(dZ, axis=1, keepdims=True)

            return grads

        #In the classification case:
        elif self.g == "clas":
            for i in range(layers,0,-1):
                #First: calculate output error.
                if i==layers:
                    dA = values["a" + str(i)] - self.Y_train.T
                    dZ = dA
                #Propagate error backwards through layers
                else:
                    dA = np.dot(params["weight" + str(i+1)].T, dZ)
                    dZ = dA @ self.act(values["a" + str(i)]).T@(1-self.act(values["a" + str(i)]))
                #Input layer
                if i==1:
                    grads["weight" + str(i)] = np.dot(dZ, self.X_train)
                    #L2 regularization (ridge regression)
                    grads["weight" + str(i)] += self.lam * params["weight" + str(i)]
                    grads["bias" + str(i)] = np.sum(dZ, axis=1, keepdims=True)
                #Hidden layer(s) if any
                else:
                    grads["weight" + str(i)] = np.dot(dZ,values["a" + str(i-1)].T)
                    #L2 regularization
                    grads["weight" + str(i)] += self.lam * params["weight" + str(i)]
                    grads["bias" + str(i)] = np.sum(dZ, axis=1, keepdims=True)
            return grads

    def update_params(self, params, grads):
        layers = len(params)//2
        params_updated = {}
        #Updating the weights and biases by the gradient descent method
        for i in range(1,layers+1):
            params_updated["weight" + str(i)] = params["weight" + str(i)] - self.eta * grads["weight" + str(i)]
            params_updated["bias" + str(i)] = params["bias" + str(i)] - self.eta * grads["bias" + str(i)]
        return params_updated

    def model(self):
        params = NeuralNet.initialize_params(self)
        #if using mini-batches:
        min = self.mini[0]

        if min == True:
            data_indices = len(self.X_train)
            #Batch size as specified in the mini list argument
            batch_size = int(self.mini[1])
            #Number of epochs as specified in the mini list argument
            epochs = int(self.mini[2])

            for k in range(epochs):
                acs = np.zeros(self.num_iters)
                for i in range(self.num_iters):
                    #Getting the accuracy/ error to check for overfitting
                    train_acc, test_acc = NeuralNet.compute_accuracy(self, params)
                    acs[i] = test_acc
                    #Creating mini-batch incdices for the design/ feature matrix and targets
                    #Might be a problem that we risk drawing same samples multiple times during one epoch. Don't know
                    chosen_datapoints = np.random.choice(data_indices, size=batch_size, replace=False)
                    #Creating a mini-batch
                    X = self.X_train[chosen_datapoints]
                    Y = self.Y_train[chosen_datapoints]

                    #Train on mini batch
                    values = NeuralNet.feed_forward(self, params, self.X_train)
                    #Propagate error
                    grads = NeuralNet.backprop(self, params, values)
                    #Update weights and biases
                    params = NeuralNet.update_params(self, params, grads)
                    #In case of overfitting. Sloppy fix. Needs replacements.
                    if i>=1 and acs[i]>acs[i-1]:
                        break
                    #Might use if i>=1 and abs(acs[i]-acs[i-1]>some_value:
                        #break
                    #Would need testing. Reluctant to introduce another adjustable parameter.

        elif min == False:
            #Training the netowrk
            for i in range(self.num_iters):
                values = NeuralNet.feed_forward(self, params, self.X_train)
                grads = NeuralNet.backprop(self, params, values)
                params = NeuralNet.update_params(self, params, grads)

        return params

    def compute_accuracy(self, params):


        values_train = NeuralNet.feed_forward(self, params, self.X_train)
        values_test = NeuralNet.feed_forward(self, params, self.X_test)

        #In the case of regression we use MSE as a measure of error
        #Might update to allow different cost functions

        if self.g == "reg":
            train_acc = np.mean((self.Y_train + values_train["a" + str(len(self.layer_sizes)-1)].T)**2)
            test_acc = np.mean((self.Y_test + values_test["a" + str(len(self.layer_sizes)-1)].T)**2)
            return train_acc, test_acc
        #In the case of classification we use percentage of correctly predicted values
        elif self.g == "clas":
            train_acc = 0
            for i in range(len(self.Y_train)):
                #Finding the true value
                true = np.argmax(self.Y_train[i])
                #Finding training value with highest probability
                pred = np.argmax(values_train["a" + str(len(layer_sizes)-1)].T[i])

                if true == pred:
                    train_acc += 1
                else:
                    continue
            #percentage of correct predictions (divided by 100)
            train_acc /= len(self.Y_train)

            #Test accuracy
            #Same as for the training error/ accuracy
            test_acc = 0
            for i in range(len(self.Y_test)):
                true = np.argmax(self.Y_test[i])
                pred = np.argmax(values_test["a" + str(len(layer_sizes)-1)].T[i])
                if true == pred:
                    test_acc += 1
                else:
                    continue
            test_acc /= len(self.Y_test)

            return train_acc,  test_acc

    def predict(self):
        params = NeuralNet.model(self, X_train, Y_train)
        prediction = NeuralNet.Feed_forward(self, X_train)
        return prediction
