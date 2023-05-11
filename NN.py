import numpy as np
import base64
from io import BytesIO
from flask import Flask
from matplotlib.figure import Figure
from neuralnet import NeuralNetMLP

app = Flask(__name__)

mnist = np.load('mnist_scaled.npz')

nn = NeuralNetMLP(n_hidden=100, l2=0.01, epochs=200, eta=0.0005, minibatch_size=100, shuffle=True, seed=1)

X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]

nn.fit(X_train=X_train[:1000], y_train=y_train[:1000], X_valid=X_train[1000:], y_valid=y_train[1000:])

y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred).astype(float) / X_test.shape[0])
print('Test accuracy: %.2f%%' % (acc * 100))

