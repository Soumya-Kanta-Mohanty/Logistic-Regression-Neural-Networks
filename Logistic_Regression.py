import math

# Dataset
x = [1, 2, 3, 4, 5, 6]
y = [0, 1, 0, 0, 1, 1]

# Initialize parameters
w = 0.0  # weight
b = 0.0  # bias
lr = 0.1  # learning rate
epochs = 1000  # number of training iterations

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Binary cross-entropy loss function
def loss_func(y_true, y_pred):
    epsilon = 1e-8  # to avoid log(0)
    return -y_true * math.log(y_pred + epsilon) - (1 - y_true) * math.log(1 - y_pred + epsilon)

# Training the logistic regression model
for epoch in range(epochs):
    dw = 0
    db = 0
    cost = 0

    for i in range(len(x)):
        z = w * x[i] + b
        y_pred = sigmoid(z)
        cost += loss_func(y[i], y_pred)

        # Compute gradients
        dw += (y_pred - y[i]) * x[i]
        db += (y_pred - y[i])

    # Update weights
    w -= lr * dw
    b -= lr * db

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {cost:.4f}")

# Prediction function
def predict(x_input):
    z = w * x_input + b
    prob = sigmoid(z)
    return 1 if prob >= 0.5 else 0  # Convert probability to binary class (0 or 1)

print("\nPredictions on training set:")
for i in range(len(x)):
    prob = sigmoid(w * x[i] + b)
    pred = predict(x[i])
    print(f"Input: {x[i]} => Predicted Probability: {prob}, Predicted Class: {pred}, Actual: {y[i]}")
