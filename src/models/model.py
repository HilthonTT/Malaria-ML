import tensorflow as tf
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import (
    Conv2D, 
    Dense, 
    InputLayer, 
    MaxPool2D, 
    Flatten, 
    BatchNormalization,
    Dropout)
from keras.metrics import (
    FalseNegatives, 
    FalsePositives, 
    TrueNegatives, 
    TruePositives, 
    Precision, 
    Recall, 
    AUC, 
    BinaryAccuracy)

from models.custom import CustomAccuracy, CustomBCE

IM_SIZE = 244
FACTOR = 1

def get_lenet_model(input_shape=(IM_SIZE, IM_SIZE, 3), dropout_rate=0, regularization_rate=0.01):
    model = Sequential()

    # Input layer
    model.add(InputLayer(input_shape=input_shape))

    # Convolutional layers
    model.add(Conv2D(filters=6,
                     kernel_size=3, 
                     strides=1, 
                     padding="valid", 
                     activation="relu", 
                     kernel_regularizer=tf.keras.regularizers.L2(regularization_rate)))
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Dropout(rate=dropout_rate))

    model.add(Conv2D(filters=16, 
                     kernel_size=3, 
                     strides=1, 
                     padding="valid", 
                     activation="relu", 
                     kernel_regularizer=tf.keras.regularizers.L2(regularization_rate)))
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=2, strides=2))

    # Fully connected layers
    model.add(Flatten())

    model.add(Dense(100, 
                    activation="relu", 
                    kernel_regularizer=tf.keras.regularizers.L2(regularization_rate)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(10, 
                    activation="relu", 
                    kernel_regularizer=tf.keras.regularizers.L2(regularization_rate)))
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    return model

def compile_model(model, learning_rate = 0.01):
    metrics = [
        TruePositives(name='tp'), 
        FalsePositives(name='fp'), 
        TrueNegatives(name='tn'), 
        FalseNegatives(name='fn'), 
        BinaryAccuracy(name='accuracy'),
        Precision(name='precision'), 
        Recall(name='recall'), 
        AUC(name='auc')
    ]

    model.compile(
        optimizer=Adam(learning_rate),
        loss=CustomBCE(FACTOR),
        metrics=CustomAccuracy(),
    )

def save_model(model):
    model.save_weights("./weights")