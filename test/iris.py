import pandas as pd
import numpy as np
import tensorflow as tf

# TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
# TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

TRAIN_URL = "~/Desktop/StockLearner/test_data/iris/iris_training.csv"
TEST_URL = "~/Desktop/StockLearner/test_data/iris/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    # train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    # test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    train_path = TRAIN_URL
    test_path = TEST_URL

    return train_path, test_path

def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size=None):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    # dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size=None):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    # assert batch_size is not None, "batch_size must not be None"
    # dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

if __name__ == "__main__":
    batch_size = 10
    (train_x, train_y), (test_x, test_y) = load_data()
    print (train_x)
    dataset = train_input_fn(train_x, train_y, batch_size)
    iterator = dataset.make_one_shot_iterator()
    # features, label = iterator.get_next()

    sess = tf.InteractiveSession()
    # print(sess.run([features, label]))
    f, l = sess.run(iterator.get_next())
    print(f)
    print("===========================")
    sf =[]
    for i in f:
        sf.append(f[i])
    print(sf)
    print("***************************")
    # stacked_f = sess.run(tf.stack(unzip(f), axis=1))
    # stacked_f = np.stack(unzip(f), axis=2)
    hf = np.stack(sf, axis=1)
    print(hf)
    print("---------------------------")
    print(l)

    a = [1, 2]
    b = [3, 4]
    c = [5, 6]
    d = np.stack([a,b,c], axis=1)
    print(d)