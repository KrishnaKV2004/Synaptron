# --- 1D Gradient Descent Example ---
def simple_quadratic_function(x):
    return x**2 - 4*x + 5  # Minimum at x = 2, f(x) = 1

def derivative_simple_quadratic(x):
    return 2*x - 4

if __name__ == "__main__":
    current_x = 0.0  # Initial guess
    learning_rate = 0.1
    epochs = 10000  # Very large number of epochs!
    threshold = 1e-6  # ✅ Small value for gradient-based early stopping

    print(f"Minimizing f(x) = x^2 - 4x + 5. Starting at x = {current_x:.2f}")
    
    for i in range(epochs):
        gradient = derivative_simple_quadratic(current_x)

        # ✅ Early stopping condition
        if abs(gradient) < threshold:
            print(f"Early stopping at epoch {i+1} — gradient {gradient:.6f} is below threshold {threshold}")
            break

        current_x = current_x - learning_rate * gradient
        loss = simple_quadratic_function(current_x)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"Epoch {i+1:2d}: x = {current_x:6.3f}, f(x) = {loss:6.3f}, gradient = {gradient:6.6f}")
    
    print(f"\nAfter {i+1} epochs, estimated minimum x = {current_x:.6f}, f(x) = {simple_quadratic_function(current_x):.6f}")
    print("The true minimum is at x = 2.0, f(x) = 1.0")