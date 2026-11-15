
import numpy as np

np.set_printoptions(precision=12, suppress=False)


# ==========================================
# STEP 5: VANISHING GRADIENT DEMONSTRATION
# ==========================================

TOTAL_LAYERS = 30
INITIAL_GRADIENT = 1.0
WEIGHT = 1.0


# ==========================================
# ACTIVATION DERIVATIVES
# ==========================================

def sigmoid_derivative(z):
    activation = 1 / (1 + np.exp(-z))
    return activation * (1 - activation)


def relu_derivative(z):
    return np.where(z > 0, 1.0, 0.0)


def tanh_derivative(z):
    activation = np.tanh(z)
    return 1 - activation ** 2


# ==========================================
# GRADIENT PROPAGATION
# ==========================================

def propagate_gradient(derivative_function, z_value):

    gradients = [INITIAL_GRADIENT]
    gradient = INITIAL_GRADIENT

    for _ in range(TOTAL_LAYERS):

        local_derivative = derivative_function(z_value)

        gradient = (
            gradient
            * WEIGHT
            * local_derivative
        )

        gradients.append(float(gradient))

    return gradients


# ==========================================
# RUN EXPERIMENTS
# ==========================================

sigmoid_gradients = propagate_gradient(
    sigmoid_derivative, 0.0
)

relu_gradients = propagate_gradient(
    relu_derivative, 1.0
)

tanh_gradients = propagate_gradient(
    tanh_derivative, 0.0
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("=" * 65)
print("VANISHING GRADIENT - ACTIVATION COMPARISON")
print("=" * 65)

print("\nControlled experiment:")
print("Total layers:", TOTAL_LAYERS)
print("Weight per layer:", WEIGHT)
print("Initial gradient:", INITIAL_GRADIENT)

print("\nGradient propagation:")
print("-" * 65)

print(
    f"{'Layer':<10}"
    f"{'Sigmoid':<22}"
    f"{'ReLU':<16}"
    f"{'Tanh':<16}"
)

print("-" * 65)

for layer in range(TOTAL_LAYERS + 1):

    print(
        f"{layer:<10}"
        f"{sigmoid_gradients[layer]:<22.12e}"
        f"{relu_gradients[layer]:<16.8f}"
        f"{tanh_gradients[layer]:<16.8f}"
    )


# ==========================================
# FINAL ANALYSIS
# ==========================================

print("\n" + "=" * 65)
print("FINAL GRADIENT ANALYSIS")
print("=" * 65)

print("\nInitial gradient:", INITIAL_GRADIENT)

print(
    "Sigmoid final gradient:",
    sigmoid_gradients[-1]
)

print(
    "ReLU final gradient:",
    relu_gradients[-1]
)

print(
    "Tanh final gradient:",
    tanh_gradients[-1]
)

print("\nGradient reduction:")

for name, gradients in [
    ("Sigmoid", sigmoid_gradients),
    ("ReLU", relu_gradients),
    ("Tanh", tanh_gradients)
]:

    reduction = (
        gradients[-1] / INITIAL_GRADIENT
    ) * 100

    print(
        f"{name}: {reduction:.12e}% "
        "of initial gradient remains"
    )


print("\n" + "=" * 65)
print("EXPERIMENT COMPLETED")
print("=" * 65)
