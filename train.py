# train.py

from model import resnet_model
from config import CONFIG
import matplotlib.pyplot as plt
from data_prepare import get_data


def training_model():
    "Function to train model with train and validation data"
    
    x_train,y_train = get_data(mode = "train")
    x_val,y_val = get_data(mode = "validation")

    history = resnet_model.fit(x_train,y_train, validation_data=(x_val,y_val), batch_size = CONFIG.BATCH_SIZE
                               epochs=CONFIG.EPOCHS)


    # Plot the chart for accuracy and loss on both training and validation
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    
    return resnet_model
