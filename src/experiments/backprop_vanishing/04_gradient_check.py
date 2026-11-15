
import numpy as np

np.set_printoptions(precision=10, suppress=True)


# ==========================================
# ACTIVATION AND LOSS
# ==========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(a):
    return a * (1 - a)


def calculate_loss(prediction, target):
    return 0.5 * np.mean((prediction - target) ** 2)


# ==========================================
# DATA AND PARAMETERS
# ==========================================

X = np.array([[0.5, 0.8]])
y = np.array([[1.0]])

W1 = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])

b1 = np.array([[0.1, 0.1]])

W2 = np.array([
    [0.5],
    [-0.4]
])

b2 = np.array([[0.2]])


# ==========================================
# FORWARD PROPAGATION
# ==========================================

def forward(X, W1, b1, W2, b2):

    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    return A1, A2


def loss_function(W1, b1, W2, b2):

    _, prediction = forward(X, W1, b1, W2, b2)

    return calculate_loss(prediction, y)


# ==========================================
# ANALYTICAL GRADIENTS
# ==========================================

def analytical_gradients():

    A1, A2 = forward(X, W1, b1, W2, b2)

    delta2 = (A2 - y) * sigmoid_derivative(A2)

    dW2 = A1.T @ delta2
    db2 = delta2

    delta1 = (delta2 @ W2.T) * sigmoid_derivative(A1)

    dW1 = X.T @ delta1
    db1 = delta1

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }


# ==========================================
# NUMERICAL GRADIENTS
# CENTRAL DIFFERENCE METHOD
# ==========================================

def numerical_gradient(parameter_name, epsilon=1e-5):

    parameter = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }[parameter_name]

    numerical = np.zeros_like(parameter)

    for index in np.ndindex(parameter.shape):

        original_value = parameter[index]

        parameter[index] = original_value + epsilon
        loss_plus = loss_function(W1, b1, W2, b2)

        parameter[index] = original_value - epsilon
        loss_minus = loss_function(W1, b1, W2, b2)

        parameter[index] = original_value

        numerical[index] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

    return numerical


# ==========================================
# GRADIENT COMPARISON
# ==========================================

analytical = analytical_gradients()

print("=" * 60)
print("NUMERICAL GRADIENT CHECKING")
print("=" * 60)

all_errors = []

for name in ["W1", "b1", "W2", "b2"]:

    numerical = numerical_gradient(name)
    manual = analytical[name]

    difference = np.abs(manual - numerical)

    relative_error = difference / np.maximum(
        1e-8,
        np.abs(manual) + np.abs(numerical)
    )

    all_errors.extend(relative_error.flatten())

    print(f"\n{name}")
    print("Analytical gradients:\n", manual)
    print("Numerical gradients:\n", numerical)
    print("Absolute difference:\n", difference)
    print("Relative error:\n", relative_error)


# ==========================================
# FINAL VERDICT
# ==========================================

max_error = max(all_errors)

print("\n" + "=" * 60)
print("FINAL GRADIENT CHECK RESULT")
print("=" * 60)

print("Maximum relative error:", max_error)

if max_error < 1e-6:
    print("EXCELLENT: Gradients match.")
elif max_error < 1e-4:
    print("GOOD: Gradients are sufficiently close.")
else:
    print("WARNING: Check the backpropagation equations.")

