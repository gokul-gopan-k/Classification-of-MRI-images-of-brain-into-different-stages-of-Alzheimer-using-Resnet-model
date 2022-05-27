import cv2
import os
from config import CONFIG
from sklearn.model_selection import train_test_split
import numpy as np

def parse_data(data_dir):
    "Function to extract image and labels"
    
    input_images = []
    input_labels = []
    
    #get names of labels from subfolder names
    labels = os.listdir(data_dir)
    
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
                #read image and alos convert from BGR to RGB form
                img_array = cv2.imread(os.path.join(path, img))[...,::-1] 
                resized_img = cv2.resize(img_array, (CONFIG.img_width, CONFIG.img_height)) 
                input_images.append(resized_img)
                input_labels.append(class_num)
                
    return input_images,input_labels

def get_data(mode):
    "Function to create train, validation and test inputs"
    "Return values as per mode is train or validation or test""
    
    data_dir_train= CONFIG.data_train
    data_dir_test = CONFIG.data_test
    
    x_train, y_train = parse_data(data_dir_train)
    test_images, test_labels = parse_data(data_dir_test)
    
    #split input test data into validationa nd test data
    x_val, x_test, y_val, y_test = train_test_split(test_images, test_labels, test_size=0.1)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    
    #return as per mode
    if mode == "train":
        return x_train, y_train
    elif mode == "Validation":
        return x_val,y_val
    else:
        return test_images, test_labels
