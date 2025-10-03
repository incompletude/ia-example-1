import numpy as np
import matplotlib.pyplot as plt
from mnist import x, y, x_test, y_test


# forward propagation


def relu(z):
    return np.maximum(z, 0)


def softmax(z):
    return np.exp(z) / sum(np.exp(z))


# def init_params():
#     w1 = np.random.rand(10, 784) - 0.5
#     b1 = np.random.rand(10, 1) - 0.5
#     w2 = np.random.rand(10, 10) - 0.5
#     b2 = np.random.rand(10, 1) - 0.5
#     return w1, b1, w2, b2


def init_params():
    w1 = np.random.rand(20, 784) - 0.5
    b1 = np.random.rand(20, 1) - 0.5
    w2 = np.random.rand(10, 20) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2


def forward_propagation(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


# backward propagation


def one_hot(y):
    one_hot_y = np.zeros((y.size, 10))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def relu_derivative(z):
    return z > 0


def backward_propagation(z1, a1, z2, a2, w1, w2, x, y):
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y  # derivative of cross-entropy + derivative of softmax (complex math simplifies to this!)
    dw2 = 1 / y.size * dz2.dot(a1.T)
    db2 = 1 / y.size * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * relu_derivative(z1)
    dw1 = 1 / y.size * dz1.dot(x.T)
    db1 = 1 / y.size * np.sum(dz1)
    return dw1, db1, dw2, db2


# gradient descent


def get_predictions(a2):
    return np.argmax(a2, 0)


def make_predictions(x, w1, b1, w2, b2):
    _, _, _, a2 = forward_propagation(w1, b1, w2, b2, x)
    predictions = get_predictions(a2)
    return predictions


def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size


def get_loss(a2, y):
    one_hot_y = one_hot(y)
    l = -np.sum(one_hot_y * np.log(a2 + 1e-8)) / y.size
    return l


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


def gradient_descent(x, y, alpha, iterations):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(a2)
            accuracy = get_accuracy(predictions, y)
            loss = get_loss(a2, y)
            print(f"iteration {i}: accuracy = {accuracy:.4f}, loss = {loss:.4f}")
    return w1, b1, w2, b2


w1, b1, w2, b2 = gradient_descent(x, y, 0.10, 5000)


# test1


def predict(index, w1, b1, w2, b2, x, y):
    x = x[:, index, None]
    y = y[index]
    prediction = make_predictions(x, w1, b1, w2, b2)
    print(f"prediction: {prediction}, label: {y}")
    image = x.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(image, interpolation="nearest")
    plt.show()


def test():
    predict(1235, w1, b1, w2, b2, x, y)
    predict(10437, w1, b1, w2, b2, x, y)
    predict(22577, w1, b1, w2, b2, x, y)
    predict(36723, w1, b1, w2, b2, x, y)

    predict(1235, w1, b1, w2, b2, x_test, y_test)
    predict(1437, w1, b1, w2, b2, x_test, y_test)
    predict(2577, w1, b1, w2, b2, x_test, y_test)
    predict(6723, w1, b1, w2, b2, x_test, y_test)

    dev_predictions = make_predictions(x_test, w1, b1, w2, b2)
    accuracy = get_accuracy(dev_predictions, y_test)
    print(f"test dataset accuracy = {accuracy:.4f}")


test()
