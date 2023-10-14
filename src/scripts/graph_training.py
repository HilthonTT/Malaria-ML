import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..")
sys.path.append(src_dir)

import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy
from data.data_initializer import get_datasets
from models.model import get_lenet_model, compile_model
from models.custom import custom_bce

FACTOR = 1

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

OPTIMIZER = Adam(learning_rate=0.01)
METRIC = BinaryAccuracy()
METRIC_VAL = BinaryAccuracy()
EPOCHS = 15

CURRENT_TIME = datetime.datetime.now().strftime("%d%m%y - %H%M%S")
CUSTOM_TRAIN_DIR = f"./logs/{CURRENT_TIME}/custom/train"
CUSTOM_VAL_DIR = f'./logs/{CURRENT_TIME}/custom/val'

@tf.function
def training_block(model, x_batch, y_batch):
    with tf.GradientTape() as recorder:
        y_pred = model(x_batch, training=True)
        loss = custom_bce(FACTOR)(y_batch, y_pred)

    partial_derivatives = recorder.gradient(loss, model.trainable_weights)
    OPTIMIZER.apply_gradients(zip(partial_derivatives, model.trainable_weights))

    return loss

@tf.function
def val_block(model, x_batch_val, y_batch_val):
    y_pred_val = model(x_batch_val, training=False)
    loss_val = custom_bce(FACTOR)(y_batch_val, y_pred_val)
    METRIC_VAL.update_state(y_batch_val, y_pred_val)

    return loss_val

def nearalearn(model, loss_function, metric, val_metric, optimizer, train_dataset, val_dataset, epochs):
    custom_train_writer = tf.summary.create_file_writer(CUSTOM_TRAIN_DIR)
    custom_val_writer = tf.summary.create_file_writer(CUSTOM_VAL_DIR)

    for epoch in range(epochs):
        print("Traning starts for epoch number {}".format(epoch+1))
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            loss = training_block(model, x_batch, y_batch)

        print("Training Loss", loss)
        print("The accuracy is", metric.result())

        with custom_train_writer.as_default():
            tf.summary.scalar('Training Loss', data=loss, step=epoch)
            tf.summary.scalar('Training Accuracy', data=metric.result(), step=epoch)

        metric.reset_states()

        for (x_batch_val, y_batch_val) in val_dataset:
            loss_val = val_block(model, x_batch_val, y_batch_val)

        print("The Validation loss", loss_val)
        print("The Validation accuracy is: ", METRIC_VAL.result())

        with custom_val_writer.as_default():
            tf.summary.scalar('Validation Loss', data=loss, step=epoch)
            tf.summary.scalar('Validation Accuracy', data=METRIC_VAL.result(), step=epoch)

        METRIC_VAL.reset_states()
        
    print("[*] Training Completed!")
    
    
def main():
    dataset, _ = tfds.load("malaria", with_info=True, as_supervised=True, split=['train'])

    train_dataset, val_dataset, test_dataset = get_datasets(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO) 

    model = get_lenet_model()
    
    nearalearn(
        model, 
        custom_bce,
        metric=METRIC, 
        val_metric=METRIC_VAL, 
        optimizer=OPTIMIZER, 
        train_dataset=train_dataset,
        val_dataset=val_dataset, 
        epochs=EPOCHS)
    
    pass
    
if __name__ == "__main__":
    main()