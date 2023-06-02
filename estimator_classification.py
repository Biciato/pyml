import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20
steps_per_epoch = np.ceil(60000 / BATCH_SIZE)

def preprocess(item):
    image = item['image']
    label = item['label']
    image = tf.image.convert_image_dtype(
    image, tf.float32)
    image = tf.reshape(image, (-1,))

    return {'image-pixels':image}, label[..., tf.newaxis]

## Step 1: Define the input functions
def train_input_fn():
    datasets = tfds.load(name='mnist')
    mnist_train = datasets['train']

    dataset = mnist_train.map(preprocess)
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.repeat()

def eval_input_fn():
    datasets = tfds.load(name='mnist')
    mnist_test = datasets['test']
    dataset = mnist_test.map(preprocess).batch(BATCH_SIZE)
    return dataset

## Step 2: feature columns
image_feature_column = tf.feature_column.numeric_column(key='image-pixels', shape=(28*28))

 ## Step 3: instantiate the estimator
dnn_classifier = tf.estimator.DNNClassifier(feature_columns=[image_feature_column], hidden_units=[32, 16], n_classes=10, model_dir='models')

## Step 4: train and evaluate
dnn_classifier.train(input_fn=train_input_fn, steps=NUM_EPOCHS * steps_per_epoch)
eval_result = dnn_classifier.evaluate(input_fn=eval_input_fn)
print(eval_result)