import numpy as np
import nnfs
from nnfs.datasets import spiral_data # We use a spiral dataset for our neural network

nnfs.init()  

# A full neural network layer (inputs, weights, biases, outputs)
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights)+self.biases
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues) #calculates dl/dw
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True) #calculates dl/db
        self.dinputs = np.dot(dvalues,self.weights.T) #calculates dl/dx

# ReLU (Rectified Linear Unit) Activation function [Used in the hidden layer]
class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output=np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation [Gives probabilities at the output layer]
class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Calculate the loss 
class Loss:
    def calculate(self, output, y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss

# Calculate loss between true probability distribution (y_true) and predicted probability (y_pred)
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape)==1:
            correct_confidences=y_pred[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidences=np.sum(y_pred_clipped*y_true,axis=1)

        negative_log_likelihoods=-np.log(correct_confidences)
        return negative_log_likelihoods
    
# Softmax + Cross-Entropy Combined Backward Propagation
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def backward(self,dvalues,y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -= 1
        self.dinputs = self.dinputs / samples


# Stochastic Gradient Descent (SGD) Optimizer - used to minimize a model's loss function by updating parameters efficiently
class Optimizer_SGD:

    def __init__(
        self,
        learning_rate=1.0,
        decay=0.,
        momentum=0.
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = \
                self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        if self.momentum:

            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights

            layer.weight_momentums = weight_updates

            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases

            layer.bias_momentums = bias_updates

        else:
            weight_updates = \
                -self.current_learning_rate * layer.dweights

            bias_updates = \
                -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1



X,y = spiral_data(samples=100, classes=3)

dense1=Layer_Dense(2,64)
activation1=Activation_ReLU()

dense2= Layer_Dense(64,3)
activation2=Activation_Softmax()

loss_function=Loss_CategoricalCrossentropy()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_SGD(
learning_rate=1.0,decay=1e-3,momentum=0.9)


# Training loop: perform forward pass, backpropagation, and parameter updates for each epoch
for epoch in range(1001):
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    loss=loss_function.calculate(activation2.output,y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if epoch % 100 == 0:
        print(
            f"epoch: {epoch}, "
            f"acc: {accuracy:.3f}, "
            f"loss: {loss:.3f}, "
            f"lr: {optimizer.current_learning_rate:.5f}"
            )

    loss_activation.backward(activation2.output, y)

    dense2.backward(loss_activation.dinputs)

    activation1.backward(dense2.dinputs)

    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    optimizer.post_update_params()
    