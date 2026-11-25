
import numpy as np


# ==========================================
# STEP 2: MULTI-LAYER FORWARD PROPAGATION
# ==========================================

np.set_printoptions(precision=4, suppress=True)


# Activation functions
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ==========================================
# INPUT LAYER
# ==========================================

X = np.array([[0.5, 0.8, 0.2]])

# ==========================================
# HIDDEN LAYER: 3 INPUTS -> 4 NEURONS
# ==========================================

W1 = np.array([
    [0.2, 0.4, 0.6, 0.3],
    [0.5, 0.1, 0.2, 0.7],
    [0.3, 0.8, 0.4, 0.2]
])

b1 = np.array([[0.1, 0.2, 0.1, 0.3]])

# Weighted sum
Z1 = np.dot(X, W1) + b1

# Activation
A1 = relu(Z1)


# ==========================================
# OUTPUT LAYER: 4 INPUTS -> 2 NEURONS
# ==========================================

W2 = np.array([
    [0.3, 0.6],
    [0.5, 0.2],
    [0.4, 0.7],
    [0.8, 0.1]
])

b2 = np.array([[0.2, 0.1]])

# Weighted sum
Z2 = np.dot(A1, W2) + b2

# Final output
A2 = sigmoid(Z2)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("=" * 50)
print("MULTI-LAYER FORWARD PROPAGATION")
print("=" * 50)

print("\nINPUT")
print("X:", X)

print("\nHIDDEN LAYER")
print("Weights W1:\n", W1)
print("Bias b1:", b1)
print("Weighted sum Z1:", Z1)
print("After ReLU A1:", A1)

print("\nOUTPUT LAYER")
print("Weights W2:\n", W2)
print("Bias b2:", b2)
print("Weighted sum Z2:", Z2)
print("Final output A2:", A2)

print("\nOutput shape:", A2.shape)
