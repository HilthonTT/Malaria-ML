from keras import Sequential
from keras.layers import Resizing, Rescaling
import tensorflow as tf
import albumentations as A

IM_SIZE = 224

transforms = A.Compose([
    A.Resize(IM_SIZE, IM_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(),
    A.RandomBrightnessContrast(p=0.2), 
])

resize_rescale_layers = Sequential([
    Resizing(IM_SIZE, IM_SIZE),
    Rescaling(1.0/255)
])

def aug_albument(image):
    data = {"image": image}
    image = transforms(**data)["image"]  

    image = tf.cast(image / 255, tf.float32)

    return image

def process_data(image, label):
    aug_img = tf.numpy_function(func=aug_albument, inp=[image], Tout=tf.float32)
    return aug_img, label

def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
    DATASET_SIZE = len(dataset)

    train_dataset = dataset.take(int(TRAIN_RATIO * DATASET_SIZE))

    val_test_dataset = dataset.skip(int(TRAIN_RATIO * DATASET_SIZE))
    val_dataset = val_test_dataset.take(int(VAL_RATIO * DATASET_SIZE))

    test_dataset = val_test_dataset.skip(int(TEST_RATIO * DATASET_SIZE))

    return train_dataset, val_dataset, test_dataset

@tf.function
def resize_rescale(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0, label

def get_datasets(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
    train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    BATCH_SIZE = 32

    train_dataset = (
        train_dataset
        .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        .map(process_data)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        val_dataset
        .map(resize_rescale)
        .batch(BATCH_SIZE)
    )

    test_dataset = (
        test_dataset
        .map(resize_rescale_layers)
        .batch(1)
    )

    return train_dataset, val_dataset, test_dataset