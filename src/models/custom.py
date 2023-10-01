from keras.losses import BinaryCrossentropy
from keras.metrics import binary_accuracy
import tensorflow as tf

def custom_bce(FACTOR):
    def loss(y_true, y_pred):
        bce = BinaryCrossentropy()
        return bce(y_true, y_pred) * FACTOR
    return loss

def custom_accuracy(FACTOR):
    def metric(y_true, y_pred):
        return binary_accuracy(y_true, y_pred) * FACTOR
    return metric

class CustomBCE(tf.keras.losses.Loss):
    def __init__(self, FACTOR):
        super(CustomBCE, self).__init__()

        self.FACTOR = FACTOR

    def call(self, y_true, y_pred):
        bce = BinaryCrossentropy()
        return bce(y_true, y_pred) * self.FACTOR
    
class CustomAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='CustomAccuracy', FACTOR=1):
        super(CustomAccuracy, self).__init__()

        self.FACTOR = FACTOR
        self.accuracy = self.add_weight(name=name, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight = None):
        output = binary_accuracy(tf.cast(y_true, dtype=tf.float32), y_pred) * self.FACTOR
        self.accuracy.assign(
            tf.math.count_nonzero(output, dtype=tf.float32) / tf.cast(len(output), dtype=tf.float32)
        )

    def result(self):
        return self.accuracy

    def reset_states(self):
        self.accuracy.assign(0.)