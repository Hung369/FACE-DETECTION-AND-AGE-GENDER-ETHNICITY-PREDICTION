import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import *
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from mtcnn import MTCNN

ETHNICITY = ['white', 'black', 'asian', 'indian', 'other']
GENDERS = ['male', 'female']

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
    model = build_model()
    model.load_weights("./model/checkpoints/best_model.weights.h5")
    face_detector = MTCNN()

    #load test iamge
    test1 = cv2.imread('./images/test1.jpg')
    img_gray = cv2.cvtColor(test1,cv2.COLOR_BGR2GRAY)
    test2 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    results = face_detector.detect_faces(test2)
    
    for res in results:
        x1,y1,width,height = res['box']
        x1,y1 = abs(x1), abs(y1)
        x2,y2 = x1+width, y1+height
        # confidence = res['confidence']

        face_reg = cv2.resize(test2[x1:x2, y1:y2], (48, 48))
        face_reg = np.expand_dims(face_reg, axis=0) 
        p = model.predict(face_reg)

        gender_pred = GENDERS[tf.where(p[2] > 0.5, 1, 0)[0][0]]
        age_pred = p[0][0].astype(np.int32)[0]
        ethnic_pred = ETHNICITY[p[1][0].argmax()]


        cv2.rectangle(test1, (x1,y1), (x2,y2), (0,255,0), thickness=2)
        cv2.putText(test1, f'Age: {age_pred} | Gender: {gender_pred} | {ethnic_pred}',(x1,y1),cv2.FONT_ITALIC,0.9,(0,0,255),2)
   

    # #display the gray image using OpenCV
    cv2.imshow('BGR Image', test1)
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
