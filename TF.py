import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from flask import Flask
from matplotlib.figure import Figure

app = Flask(__name__)


X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])
X_train_norm =  (X_train - np.mean(X_train))/np.std(X_train)
ds_train_orig = tf.data.Dataset.from_tensor_slices((tf.cast(X_train_norm, tf.float32), tf.cast(y_train, tf.float32)))

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = tf.Variable(0.0, name='weight')
        self.b = tf.Variable(0.0, name='bias')

    def call(self, x):
        return self.w * x + self.b
    
tf.random.set_seed(1)

model = MyModel()
model.build(input_shape=(None, 1))
model.summary()

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model(inputs), outputs)
    dW, db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


num_epochs = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1
steps_per_epoch = int(np.ceil(len(y_train) / batch_size))

ds_train = ds_train_orig.shuffle(buffer_size=len(y_train))
ds_train = ds_train.repeat(count=None)
ds_train = ds_train.batch(1)
Ws, bs = [], []

for i, batch in enumerate(ds_train):
    if i >= steps_per_epoch * num_epochs:
        # break the infinite loop
        break
    Ws.append(model.w.numpy())
    bs.append(model.b.numpy())
    bx, by = batch
    loss_val = loss_fn(model(bx), by)
    train(model, bx, by, learning_rate=learning_rate)


X_test = np.linspace(0, 9, num=100).reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))

@app.route('/')
def index():   
    fig = Figure()
    fig.set_size_inches(13, 5)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(X_train_norm, y_train, 'o', markersize=10)
    ax.plot(X_test_norm, y_pred, '--', lw=3)
    ax.legend(['Training examples', 'Linear Reg'], fontsize=15)
    ax.set_xlabel('x', size=15)
    ax.set_ylabel('y', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(Ws, lw=3)
    ax.plot(bs, lw=3)
    ax.legend(['Weight w', 'Bias unit b'], fontsize=15)
    ax.set_xlabel('Iteration', size=15)
    ax.set_ylabel('Value', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"