import os, gzip, struct, urllib.request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# ---------------- OLD: MNIST-style loader (kept) ----------------
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

# ---------------- NEW: Fashion-style PNG loader ----------------
def load_my_fashion_png(path, invert_if_needed=False):
    """
    Returns:
      x: (1, 784) float32 normalized
      img28: (28, 28) float32 normalized for display
    """
    img = Image.open(path).convert("L")

    # Optional invert if your foreground/background are reversed
    if invert_if_needed:
        img = ImageOps.invert(img)

    # Resize to Fashion-MNIST resolution
    img = img.resize((28, 28), resample=Image.Resampling.BILINEAR)

    # Convert to numpy and normalize
    arr = np.array(img).astype(np.float32) / 255.0

    # Flatten for dense net
    x = arr.reshape(1, 784)
    return x, arr

# -------- Fashion-MNIST loader (downloads once) --------
FASHION_DIR = "fashion_mnist_data"
BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def download_fashion_mnist():
    os.makedirs(FASHION_DIR, exist_ok=True)
    for fname in FILES.values():
        path = os.path.join(FASHION_DIR, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(f"{BASE_URL}/{fname}", path)

def load_idx_images(gz_path):
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0  # (n, 784)

def load_idx_labels(gz_path):
    with gzip.open(gz_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)

def load_fashion_mnist():
    download_fashion_mnist()
    X_train = load_idx_images(os.path.join(FASHION_DIR, FILES["train_images"]))
    y_train = load_idx_labels(os.path.join(FASHION_DIR, FILES["train_labels"]))
    X_test  = load_idx_images(os.path.join(FASHION_DIR, FILES["test_images"]))
    y_test  = load_idx_labels(os.path.join(FASHION_DIR, FILES["test_labels"]))
    return X_train, y_train, X_test, y_test

# ---------- Load Fashion-MNIST ----------
X_train, y_train, X_test, y_test = load_fashion_mnist()
print(X_test.shape, y_test.shape)  # (10000, 784) (10000,)

# ---------- Define network pieces (forward only) ----------
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

# ---------- SAME architecture as training ----------
layer1 = Layer_Dense(784, 128)
layer2 = Layer_Dense(128, 128)
layer3 = Layer_Dense(128, 64)
layer4 = Layer_Dense(64, 10)

act1 = Activation_ReLU()
act2 = Activation_ReLU()
act3 = Activation_ReLU()
softmax = Activation_SoftMax()

# ---------- Load FASHION weights ----------
data = np.load("Fmnist_model_weights.npz")   # make sure this file exists
layer1.weights = data["layer1_W"]; layer1.biases = data["layer1_b"]
layer2.weights = data["layer2_W"]; layer2.biases = data["layer2_b"]
layer3.weights = data["layer3_W"]; layer3.biases = data["layer3_b"]
layer4.weights = data["layer4_W"]; layer4.biases = data["layer4_b"]

# ============================================================
# OLD: Predict one Fashion-MNIST test sample (kept, commented)
# ============================================================
# idx = 9  # change this 0..9999
# x = X_test[idx:idx+1]     # (1, 784)
# true_label = int(y_test[idx])
#
# layer1.forward(x); act1.forward(layer1.output)
# layer2.forward(act1.output); act2.forward(layer2.output)
# layer3.forward(act2.output); act3.forward(layer3.output)
# layer4.forward(act3.output); softmax.forward(layer4.output)
#
# probs = softmax.output[0]
# pred = int(np.argmax(probs))
#
# print("True:", true_label, "-", LABELS[true_label])
# print("Pred:", pred, "-", LABELS[pred])
# print("Top prob:", float(np.max(probs)))
# print("Probabilities:", probs)
#
# plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
# plt.title(f"True={LABELS[true_label]} | Pred={LABELS[pred]}")
# plt.axis("off")
# plt.show()

# ============================================================
# NEW: Predict from your own PNG file
# ============================================================
# Put your image file next to this script.
# If your PNG is dark item on light background: invert_if_needed=False
# If your PNG is light item on dark background: invert_if_needed=True
x, img28 = load_my_fashion_png("my_fashion.png", invert_if_needed=False)

layer1.forward(x); act1.forward(layer1.output)
layer2.forward(act1.output); act2.forward(layer2.output)
layer3.forward(act2.output); act3.forward(layer3.output)
layer4.forward(act3.output); softmax.forward(layer4.output)

probs = softmax.output[0]
pred = int(np.argmax(probs))

print("Pred:", pred, "-", LABELS[pred])
print("Top prob:", float(np.max(probs)))
print("Probabilities:", probs)

# Show the 28x28 image the model actually used
plt.imshow(img28, cmap="gray")
plt.title(f"Pred={LABELS[pred]}")
plt.axis("off")
plt.show()