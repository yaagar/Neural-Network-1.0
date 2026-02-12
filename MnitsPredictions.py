import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def load_my_digit(path, invert_if_needed=False):
    # 1) open image, convert to grayscale
    img = Image.open(path).convert("L")

    # 2) optional invert (set True if you drew black digit on white background)
    if invert_if_needed:
        img = ImageOps.invert(img)

    # 3) resize to 28x28 like MNIST
    img = img.resize((28, 28))

    # 4) convert to numpy, normalize to 0..1
    arr = np.array(img).astype(np.float32) / 255.0

    # 5) flatten to (1, 784)
    x = arr.reshape(1, 784)

    return x, arr

# ---------- Load MNIST (not required for prediction, but kept here if you want to compare later) ----------
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = np.asarray(X).astype(np.float32) / 255.0
y = np.asarray(y).astype(np.int64)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# ---------- Define network pieces (only forward needed) ----------
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons).astype(np.float32)
        self.biases = np.zeros((1, n_neurons), dtype=np.float32)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# ---------- Build SAME architecture as training ----------
layer1 = Layer_Dense(784, 128)
layer2 = Layer_Dense(128, 128)
layer3 = Layer_Dense(128, 64)
layer4 = Layer_Dense(64, 10)

act1 = Activation_ReLU()
act2 = Activation_ReLU()
act3 = Activation_ReLU()
softmax = Activation_SoftMax()

# ---------- Load saved weights ----------
data = np.load("mnist_model_weights.npz")
layer1.weights = data["layer1_W"]; layer1.biases = data["layer1_b"]
layer2.weights = data["layer2_W"]; layer2.biases = data["layer2_b"]
layer3.weights = data["layer3_W"]; layer3.biases = data["layer3_b"]
layer4.weights = data["layer4_W"]; layer4.biases = data["layer4_b"]

# ---------- YOUR INPUT IMAGE ----------
# Put my_digit.png in the same folder as this script.
# If you drew WHITE digit on BLACK background -> invert_if_needed=False
# If you drew BLACK digit on WHITE background -> invert_if_needed=True
x, img28 = load_my_digit("my_digit.png", invert_if_needed=False)

# ---------- Forward pass ----------
layer1.forward(x); act1.forward(layer1.output)
layer2.forward(act1.output); act2.forward(layer2.output)
layer3.forward(act2.output); act3.forward(layer3.output)
layer4.forward(act3.output); softmax.forward(layer4.output)

probs = softmax.output[0]
pred = int(np.argmax(probs))

print("Predicted:", pred)
print("Top prob:", float(np.max(probs)))
print("Probabilities:", probs)

# Show the 28x28 image the model actually used
confidence = float(np.max(probs)) * 100

plt.figure(figsize=(5,5))
plt.imshow(img28, cmap="gray")

plt.title(
    f"Handwritten Digit Recognition\n"
    f"Prediction: {pred}  |  Confidence: {confidence:.2f}%",
    fontsize=12,
    fontweight="bold"
)

plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
