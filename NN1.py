import sys
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
import time

time_seed = int(time.time())
nnfs.init()
momentum = 0.9
X, y = spiral_data(300,5)
np.random.seed(time_seed)
N = 200
batch_size = 100
learningRate = 0.1
#X = np.random.randn(N, 3)
#y = (X[:, 0] + X[:, 1] > 0).astype(int)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs                
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        if not hasattr(self, "weight_momentum"):
            self.weight_momentum = np.zeros_like(self.weights)
            self.bias_momentum   = np.zeros_like(self.biases)
        
        dw = np.dot(self.inputs.T, dvalues) / self.inputs.shape[0]
        db  = np.sum(dvalues, axis=0, keepdims=True) / self.inputs.shape[0]
        # v = mu*v - lr*grad
        self.weight_momentum = - learningRate * dw + momentum * self.weight_momentum
        self.bias_momentum   = - learningRate * db + momentum * self.bias_momentum   
        
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.weights += self.weight_momentum
        self.biases  += self.bias_momentum
        pass

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        self.output = exp_values/np.sum(exp_values,axis=1,keepdims = True)
    pass
    def backward(self, y_true):
        samples = len(self.output)
        Y = np.eye(self.output.shape[1])[y_true]
        self.dinputs = (self.output - Y)

class Loss:
    def calculate(self, output, y):
        samples_losses = self.forward(output, y)
        data_loss = np.mean(samples_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) ==2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
        
        negative_log_likelyhoods = -np.log(correct_confidences)
        return negative_log_likelyhoods

layer1 = Layer_Dense(2,64)
layer2 = Layer_Dense(64,64)
layer3 = Layer_Dense(64,64)
layer4 = Layer_Dense(64,5)
activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_ReLU()
activation4 = Activation_SoftMax()
loss_function = Loss_CategoricalCrossentropy()
for epoch in range(2000):
        randomIndex = np.random.randint(0,300-batch_size)
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
    #for start in range(0, len(X), batch_size):
        #end = start + batch_size
        #X_batch = X_shuffled[start:end]
        #y_batch = y_shuffled[start:end]
        end = randomIndex + batch_size
        X_batch = X_shuffled[randomIndex:end]
        y_batch = y_shuffled[randomIndex:end]
        layer1.forward(X_batch)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        layer3.forward(activation2.output)
        activation3.forward(layer3.output)
        layer4.forward(activation3.output)
        activation4.forward(layer4.output)
        loss = loss_function.calculate(activation4.output, y_batch)
        activation4.backward(y_batch)
        layer4.backward(activation4.dinputs)
        activation3.backward(layer4.dinputs)
        layer3.backward(activation3.dinputs)
        activation2.backward(layer3.dinputs)
        layer2.backward(activation2.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)
        
    # Evaluate/print/early-stop on full dataset (not last batch)
        if epoch % 10 == 0:
            layer1.forward(X)
            activation1.forward(layer1.output)
            layer2.forward(activation1.output)
            activation2.forward(layer2.output)
            layer3.forward(activation2.output)
            activation3.forward(layer3.output)
            layer4.forward(activation3.output)
            activation4.forward(layer4.output)
            if epoch>1:
                loss_old = loss_full
                loss_full = loss_function.calculate(activation4.output, y)
                if loss_old>loss_full:
                    learningRate = learningRate*0.999
                else:
                    learningRate = learningRate*1.0001
            else:
                loss_full = loss_function.calculate(activation4.output, y)
            pred_full = np.argmax(activation4.output, axis=1)
            acc_full = np.mean(pred_full == y)
            if epoch%1000 ==0:
                print(loss_full, acc_full*100)

            if acc_full == 1 and loss_full < 0.01:
                print(loss_full, acc_full)
                print("hello")
                break

print(learningRate)

# Full forward pass before plotting predictions
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)
activation3.forward(layer3.output)
layer4.forward(activation3.output)
activation4.forward(layer4.output)
pred = np.argmax(activation4.output, axis=1)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=y, s=15, cmap='coolwarm')

plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=pred, s=15, cmap='coolwarm')

plt.show()