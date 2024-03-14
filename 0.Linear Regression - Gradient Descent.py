"""
Linear Regression Using Gradient Descent

1. Generate data using numpy
2. Linear regression, y = wx + b
3. Compute loss, l = (y - y_hat)**2/n
    - y -> actual value
    - y_hat -> predicted value
    - n -> number of samples
4. Update weight and bias using gradient descent
    - dldw -> partial derivative of loss l wrt w
        - 2*(y - (wx + b))*(-x)
    - dldb -> partial derivative of loss l wrt b
        - 2*(y - (wx + b))*(-1)
    Update eq.
    w_new = w - learning_rate*dldw
    b_new = b - learning_rate*dldb

5. Run step 2, 3, 4 for N epochs

"""
import numpy as np

X = np.random.rand(100)
Y = 5.15 * X + 9


# initialise w and b
def initialise_weights():
    w = np.random.rand()
    b = np.random.rand()
    return w, b


def predict(X, w, b):
    return w * X + b


def compute_loss(Y, y_hat):
    return np.sum((Y - y_hat) ** 2 / Y.shape[0], axis=0)


def grad_descent(Y, y_hat, w, b):
    # 2 * (y - (wx + b)) * (-x)
    dldw = np.sum(2 * (Y - y_hat) * (-X), axis=0) / Y.shape[0]
    dldb = np.sum(2 * (Y - y_hat) * (-1), axis=0) / Y.shape[0]

    return dldw, dldb


w, b = initialise_weights()
learning_rate = 0.01
epochs = 1000
for epoch in range(epochs):
    y_hat = predict(X, w, b)
    dldw, dldb = grad_descent(Y, y_hat, w, b)
    w = w - learning_rate * dldw
    b = b - learning_rate * dldb
    loss = compute_loss(Y, y_hat)
    print(f'Loss : {loss}, w : {w}, b : {b}')
