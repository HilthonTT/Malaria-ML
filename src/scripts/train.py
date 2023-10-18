import datetime
import os
import sys


script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..")
sys.path.append(src_dir)

import tensorflow_datasets as tfds
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from visualization.visualization import show_image, show_loss
from data.data_initializer import get_datasets
from models.model import get_lenet_model, compile_model, save_model
from models.custom import LogImagesCallback

CURRENT_TIME = datetime.datetime.now().strftime("%d%m%y - %H%M%S")
METRIC_DIR = './logs/' + CURRENT_TIME + "/metrics"
LOG_DIR = './logs/' + CURRENT_TIME

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

EPOCHS = 15

def determine_parasite(x):
    if ( x < 0.5):
        return str("Parasitized")
    return str("Uninfected")

def show_first_image(dataset):
    for batch in dataset.take(1): 
        images, labels = batch 
        image_shape = images[0].shape 
        show_image(images[0])
        print("Size of one image:", image_shape)
        
def scheduler(epoch, lr):
    train_writer = tf.summary.create_file_writer(METRIC_DIR)
    if epoch <= 1:
        learning_rate = lr
    else:
        learning_rate = lr * tf.math.exp(-0.1)
        learning_rate = learning_rate.numpy()

    with train_writer.as_default():
        tf.summary.scalar("Learning Rate", data=learning_rate, step=epoch)
        
    return learning_rate
        
def main():
    dataset, dataset_info = tfds.load("malaria", with_info=True, as_supervised=True, split=['train'])

    train_dataset, val_dataset, test_dataset = get_datasets(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO) 

    show_first_image(val_dataset)

    model = get_lenet_model()
    compile_model(model)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard("./logs")
    scheduler_callback = LearningRateScheduler(scheduler, verbose=1)
    image_callback = LogImagesCallback(test_dataset, model)

    history = model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=EPOCHS, 
        verbose=1,
        callbacks=[tensorboard_callback, scheduler_callback, image_callback])
    
    print(history.history)
    
    save_model(model)
    
    show_loss(history)
    
if __name__ == "__main__":
    main()