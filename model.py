import tensorflow as tf
from config import CONFIG
from tensorflow.keras.models import Sequential

def create_model():
    "Function to create mode with Resnet50 model as backbone"
    
    model_base = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(CONFIG.img_width, CONFIG.img_height, 3))
    
    #freeze layers
    for layer in model_base.layers:
            layer.trainable=False
            
    resnet_model = Sequential()
    resnet_model.add(model_base)
    resnet_model.add(tf.keras.layers.Flatten())
    resnet_model.add(tf.keras.layers.Dense(512, activation='relu'))
    resnet_model.add(tf.keras.layers.Dense(4, activation='softmax'))
    
    resnet_model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return resnet_model
