import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..")
sys.path.append(src_dir)

import tensorflow_datasets as tfds
import tensorflow as tf
from visualization.visualization import show_image, show_loss
from data.data_initializer import get_datasets
from models.model import get_lenet_model, compile_model, save_model
from models.custom import custom_bce

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

EPOCHS = 20

def main():
    dataset, _ = tfds.load("malaria", with_info=True, as_supervised=True, split=['train'])

    train_dataset, val_dataset, test_dataset = get_datasets(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO) 

    for batch in train_dataset.take(1): 
        images, labels = batch 
        image_shape = images[0].shape 
        show_image(images[0])
        print("Size of one image:", image_shape)

    model = get_lenet_model()
    compile_model(model)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard("./logs")

    history = model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=EPOCHS, 
        verbose=1,
        callbacks=[tensorboard_callback])
    
    save_model(model)
    
    show_loss(history)
    
if __name__ == "__main__":
    main()