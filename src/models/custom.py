from keras.losses import BinaryCrossentropy
from keras.metrics import binary_accuracy
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import io
import datetime

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

    def reset_state(self):
        self.accuracy.assign(0.)
        
class LogImagesCallback(Callback):
    def __init__(self, test_dataset, model):
        super().__init__()
        
        self.test_dataset = test_dataset
        self.model = model
        
    def on_epoch_end(self, epoch, logs=None):
        labels = []
        inp = []
        
        for x, y in self.test_dataset.as_numpy_iterator():
            labels.append(y)
            inp.append(x)
            
        labels = np.array([i[0] for i in labels])
        predicted = self.model.predict(np.array(inp)[:,0,...])
        
        threshold = 0.5
        
        cm = confusion_matrix(labels, predicted > threshold)
        
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True)
        plt.title("Confusion matric - {}".format(threshold))
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.axis("off")
           
        buffer = io.BytesIO()
        plt.savefig(buffer, format = 'png')
        
        image = tf.image.decode_png(buffer.getvalue(), channels=3)
        
        current_time = datetime.datetime.now().strftime("%d%m%y - %H%M%S")
        image_dir = './logs/' + current_time + "/images"
        image_writer = tf.summary.create_file_writer(image_dir)
        
        with image_writer.as_default():
            tf.summary.image("Training data", image, step=epoch)
        
        plt.show()
        
        