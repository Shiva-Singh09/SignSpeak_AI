
import numpy as np

np.set_printoptions(precision=8, suppress=True)

# ==========================================
# CONFIGURATION
# ==========================================

LAYERS = 20
NEURONS = 32
SEED = 42


# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1 - a)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


# ==========================================
# INITIALIZATION METHODS
# ==========================================

def initialize_weights(method, rng):

    weights = []

    for _ in range(LAYERS):

        if method == "small":
            std = 0.01

        elif method == "xavier":
            std = np.sqrt(1 / NEURONS)

        elif method == "he":
            std = np.sqrt(2 / NEURONS)

        elif method == "large":
            std = 1.0

        matrix = rng.normal(
            0, std, (NEURONS, NEURONS)
        )

        weights.append(matrix)

    return weights


# ==========================================
# GRADIENT FLOW EXPERIMENT
# ==========================================

def analyze_network(name, activation, derivative,
                    initialization):

    rng = np.random.default_rng(SEED)

    weights = initialize_weights(
        initialization, rng
    )

    x = rng.normal(0, 1, (1, NEURONS))

    z_values = []
    activations = [x]

    # Forward propagation
    for w in weights:

        z = activations[-1] @ w
        a = activation(z)

        z_values.append(z)
        activations.append(a)

    # Backward propagation
    grad = np.ones((1, NEURONS)) / np.sqrt(NEURONS)

    gradient_norms = []

    for layer in reversed(range(LAYERS)):

        delta = grad * derivative(z_values[layer])

        gradient_norms.append(
            np.linalg.norm(delta)
        )

        grad = delta @ weights[layer].T

    gradient_norms.reverse()

    return {
        "name": name,
        "initialization": initialization,
        "first_layer_gradient": gradient_norms[0],
        "last_layer_gradient": gradient_norms[-1],
        "gradient_norms": gradient_norms
    }


# ==========================================
# RUN COMPARISONS
# ==========================================

experiments = [
    ("Sigmoid", sigmoid, sigmoid_derivative, "small"),
    ("Sigmoid", sigmoid, sigmoid_derivative, "xavier"),
    ("Tanh", tanh, tanh_derivative, "small"),
    ("Tanh", tanh, tanh_derivative, "xavier"),
    ("ReLU", relu, relu_derivative, "small"),
    ("ReLU", relu, relu_derivative, "he"),
    ("ReLU", relu, relu_derivative, "large"),
]

results = []

for name, activation, derivative, initialization in experiments:

    result = analyze_network(
        name,
        activation,
        derivative,
        initialization
    )

    results.append(result)


# ==========================================
# DISPLAY INITIALIZATION RESULTS
# ==========================================

print("=" * 75)
print("WEIGHT INITIALIZATION COMPARISON")
print("=" * 75)

print(
    f"{'Activation':<15}"
    f"{'Initialization':<18}"
    f"{'Layer 1 Gradient':<22}"
    f"{'Layer 20 Gradient':<22}"
)

print("-" * 75)

for result in results:

    print(
        f"{result['name']:<15}"
        f"{result['initialization']:<18}"
        f"{result['first_layer_gradient']:<22.8e}"
        f"{result['last_layer_gradient']:<22.8e}"
    )


# ==========================================
# GRADIENT CLIPPING
# ==========================================

print("\n" + "=" * 75)
print("GRADIENT CLIPPING DEMONSTRATION")
print("=" * 75)

# Artificially large gradients for demonstration
gradients = [
    np.array([[30.0, 40.0]]),
    np.array([[60.0, 80.0]])
]

# Calculate global gradient norm
global_norm = np.sqrt(
    sum(np.sum(g ** 2) for g in gradients)
)

max_norm = 5.0

print("\nOriginal gradients:")
for g in gradients:
    print(g)

print("\nOriginal global norm:", global_norm)
print("Maximum allowed norm:", max_norm)

# Clip only if global norm exceeds maximum
if global_norm > max_norm:

    scale = max_norm / (global_norm + 1e-12)

    clipped_gradients = [
        g * scale for g in gradients
    ]

else:
    clipped_gradients = gradients

clipped_norm = np.sqrt(
    sum(np.sum(g ** 2) for g in clipped_gradients)
)

print("\nClipped gradients:")
for g in clipped_gradients:
    print(g)

print("\nClipped global norm:", clipped_norm)

print("\nGradient clipping successful:",
      clipped_norm <= max_norm + 1e-9)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 75)
print("EXPERIMENT COMPLETED")
print("=" * 75)
