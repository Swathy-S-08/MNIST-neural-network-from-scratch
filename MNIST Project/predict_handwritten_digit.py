import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from Neural_structure import *

# Create network
dense1 = Layer_Dense(784, 128)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(128, 10)
activation2 = Activation_Softmax()

# Load trained weights
dense1.weights = np.load("MNIST Project/weights_biases/dense1_weights.npy")
dense1.biases = np.load("MNIST Project/weights_biases/dense1_biases.npy")

dense2.weights = np.load("MNIST Project/weights_biases/dense2_weights.npy")
dense2.biases = np.load("MNIST Project/weights_biases/dense2_biases.npy")

# Load image
image = Image.open("MNIST Project/handwritten_digit.png")

# Convert to grayscale
image = image.convert("L")

# Resize to MNIST size
image = image.resize((28, 28))

# Convert to numpy array
image = np.array(image)

# Invert colors if needed
image = 255 - image

# Normalize
image = image.astype(np.float32) / 255.0

# Flatten
image = image.reshape(1, 784)

# Forward pass
dense1.forward(image)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Prediction
prediction = np.argmax(activation2.output, axis=1)

# Confidence
confidence = np.max(activation2.output)

print(f"Predicted Digit: {prediction[0]}")
print(f"Confidence: {confidence:.4f}")

plt.imshow(image.reshape(28,28), cmap="gray")

plt.title(
    f"Predicted: {prediction[0]} | "
    f"Confidence: {confidence:.2%}"
)

plt.show()