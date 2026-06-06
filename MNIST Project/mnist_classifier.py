import os
import numpy as np
import matplotlib.pyplot as plt
from Neural_structure import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_mnist_images(filename):

    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)

    data = data[16:]  # skip header

    images = data.reshape(-1, 28, 28)

    return images


def load_mnist_labels(filename):

    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)

    return data[8:]  # skip header

# Load the dataset

X_train = load_mnist_images(os.path.join(BASE_DIR,"Dataset","train-images-idx3-ubyte","train-images.idx3-ubyte"))

y_train = load_mnist_labels(os.path.join(BASE_DIR,"Dataset","train-labels-idx1-ubyte","train-labels.idx1-ubyte"))

X_test = load_mnist_images(os.path.join(BASE_DIR,"Dataset","t10k-images-idx3-ubyte","t10k-images.idx3-ubyte"))

y_test = load_mnist_labels(os.path.join(BASE_DIR,"Dataset","t10k-labels-idx1-ubyte","t10k-labels.idx1-ubyte"))

X_test_original = X_test.copy() # Save original images

# Flatten images
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize pixel values
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

X = X_train
y = y_train

batch_size = 128 # Mini batch training

# Training loop
dense1 = Layer_Dense(784,128)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(128,10)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(learning_rate=0.001)

for epoch in range(31):
    for step in range(0,len(X),batch_size):
        indices = np.random.permutation(len(X))
        batch_X = X[step:step + batch_size]
        batch_y = y[step:step + batch_size]

        dense1.forward(batch_X)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss = loss_function.calculate(activation2.output,batch_y)

        predictions = np.argmax(activation2.output,axis=1)

        accuracy = np.mean(predictions == batch_y)

        if epoch % 10 == 0 and step + batch_size >= len(X):
            print(
                f"epoch: {epoch}, "
                f"acc: {accuracy:.3f}, "
                f"loss: {loss:.3f}"
            )

        loss_activation.backward(activation2.output,batch_y)

        dense2.backward(loss_activation.dinputs)

        activation1.backward(dense2.dinputs)

        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()

        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

        optimizer.post_update_params()

# Save trained parameters
np.save("MNIST Project/weights_biases/dense1_weights.npy", dense1.weights)
np.save("MNIST Project/weights_biases/dense1_biases.npy", dense1.biases)

np.save("MNIST Project/weights_biases/dense2_weights.npy", dense2.weights)
np.save("MNIST Project/weights_biases/dense2_biases.npy", dense2.biases)

print("Model parameters saved.\n")


# Testing loop
dense1.forward(X_test)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output,y_test)

predictions = np.argmax(activation2.output,axis=1)

accuracy = np.mean(predictions == y_test)

print(f"\n Test Accuracy: {accuracy:.3f}, "f"Test Loss: {loss:.3f}")

# Visualization
sample = np.random.randint(0, len(X_test))

dense1.forward(X_test[sample:sample+1])
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

prediction = np.argmax(activation2.output)

plt.imshow(X_test_original[sample], cmap='gray')

plt.title(
    f"Actual: {y_test[sample]} | Predicted: {prediction}"
)

plt.show()