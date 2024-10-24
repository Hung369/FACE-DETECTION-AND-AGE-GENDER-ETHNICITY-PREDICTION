import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import *
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from DataGenerator import CustomDataGen

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

EPOCHS = 70
BATCH_SIZE = 64

class PreprocessLayer(Layer):
    def call(self, inputs):
        return tf.keras.applications.vgg16.preprocess_input(tf.cast(inputs, tf.float32))

def build_model(input_shape=(48, 48, 3)):
    # Input layer with uint8 data type, casting to float32
    i = tf.keras.layers.Input(input_shape, dtype=tf.uint8)
    x = PreprocessLayer()(i)

    # VGG16 backbone model
    backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=x)
    output_layer = backbone.get_layer("block5_conv3").output

    # Define branches for multi-output
    def build_age_branch(input_tensor):
        x = tf.keras.layers.Dense(1024)(input_tensor)
        x = LeakyReLU(alpha=0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation=None, name='age_output')(x)
        return x

    def build_ethnicity_branch(input_tensor):
        x = tf.keras.layers.Dense(500)(input_tensor)
        x = LeakyReLU(alpha=0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(5, activation='softmax', name='ethnicity_output')(x)
        return x

    def build_gender_branch(input_tensor):
        x = tf.keras.layers.Dense(500)(input_tensor)
        x = LeakyReLU(alpha=0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='gender_output')(x)
        return x

    # Flatten the backbone output and create branches
    x = tf.keras.layers.Flatten()(output_layer)
    output_age = build_age_branch(x)
    output_ethnicity = build_ethnicity_branch(x)
    output_gender = build_gender_branch(x)

    # Define the final multi-output model
    model = Model(i, [output_age, output_ethnicity, output_gender])

    return model



if __name__ == "__main__":
    data = pd.read_csv('./data/age_gender.csv')
    train, val = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = CustomDataGen(train, data_augmentation = True)
    val_dataset = CustomDataGen(val)
    model = build_model()
    model.summary()
    plot_model(model, to_file='./model/model_plot.png', show_shapes=True, show_layer_names=True)

    model.compile(Adam(learning_rate=1e-4), loss = ['mse', 'categorical_crossentropy', 'binary_crossentropy'],
                    loss_weights = [0.005,0.8,0.8], # focal loss (add weights)
                    metrics = {'age_output': 'mean_absolute_error', 'ethnicity_output': 'accuracy', 'gender_output': 'accuracy'})

    # Reduce learning rate when a metric has stopped improving. 
    plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.3, patience = 2)

    # Early stopping (stops training when validation doesn't improve for {patience} epochs)
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5)

    # Saves the best version of the model to disk (as measured on the validation data set)
    save_best = tf.keras.callbacks.ModelCheckpoint(filepath='./model/checkpoints/best_model.weights.h5', monitor='val_loss', 
                                                    save_best_only=True, mode='min', save_weights_only=True)

    history = model.fit(train_dataset, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data = val_dataset, callbacks = [es, save_best, plateau])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.yscale('log')
    plt.show()
