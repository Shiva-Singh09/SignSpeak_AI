
import numpy as np

np.set_printoptions(precision=6, suppress=True)


# ==========================================
# ACTIVATION FUNCTION
# ==========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(activation):
    # Derivative using sigmoid's output
    return activation * (1 - activation)


# ==========================================
# INITIAL NETWORK PARAMETERS
# ==========================================

# 1 sample, 2 input features
X = np.array([[0.5, 0.8]])

# Target output
y = np.array([[1.0]])

# Input -> Hidden (2 -> 2)
W1 = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])

b1 = np.array([[0.1, 0.1]])

# Hidden -> Output (2 -> 1)
W2 = np.array([
    [0.5],
    [-0.4]
])

b2 = np.array([[0.2]])

learning_rate = 0.1


# ==========================================
# FORWARD PROPAGATION
# ==========================================

def forward_pass(X, W1, b1, W2, b2):

    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2


# ==========================================
# LOSS FUNCTION
# ==========================================

def calculate_loss(prediction, target):
    return 0.5 * np.mean((prediction - target) ** 2)


# ==========================================
# STEP 1: FORWARD PASS
# ==========================================

Z1, A1, Z2, A2 = forward_pass(
    X, W1, b1, W2, b2
)

loss_before = calculate_loss(A2, y)


# ==========================================
# STEP 2: BACKWARD PASS
# ==========================================

# Output layer error
delta2 = (A2 - y) * sigmoid_derivative(A2)

# Output layer gradients
dW2 = A1.T @ delta2
db2 = np.sum(delta2, axis=0, keepdims=True)

# Hidden layer error
delta1 = (delta2 @ W2.T) * sigmoid_derivative(A1)

# Hidden layer gradients
dW1 = X.T @ delta1
db1 = np.sum(delta1, axis=0, keepdims=True)


# ==========================================
# STEP 3: UPDATE PARAMETERS
# ==========================================

W1_old = W1.copy()
W2_old = W2.copy()

W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1

W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2


# ==========================================
# STEP 4: CHECK UPDATED NETWORK
# ==========================================

_, _, _, updated_prediction = forward_pass(
    X, W1, b1, W2, b2
)

loss_after = calculate_loss(updated_prediction, y)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("=" * 55)
print("MANUAL BACKPROPAGATION - NUMPY")
print("=" * 55)

print("\n1. FORWARD PASS")
print("Input:", X)
print("Target:", y)
print("Hidden weighted sum Z1:", Z1)
print("Hidden activation A1:", A1)
print("Output weighted sum Z2:", Z2)
print("Prediction:", A2)

print("\n2. LOSS")
print("Loss before update:", loss_before)

print("\n3. BACKWARD PASS")
print("Output delta:", delta2)
print("Hidden delta:", delta1)

print("\nOutput weight gradients dW2:\n", dW2)
print("Output bias gradients db2:", db2)

print("\nHidden weight gradients dW1:\n", dW1)
print("Hidden bias gradients db1:", db1)

print("\n4. WEIGHT UPDATES")
print("W1 before:\n", W1_old)
print("W1 after:\n", W1)

print("\nW2 before:\n", W2_old)
print("W2 after:\n", W2)

print("\n5. LEARNING RESULT")
print("Prediction before:", A2)
print("Prediction after:", updated_prediction)
print("Loss before:", loss_before)
print("Loss after:", loss_after)

if loss_after < loss_before:
    print("\nSUCCESS: Loss decreased after one update.")
else:
    print("\nLoss did not decrease. Inspect the gradients.")
