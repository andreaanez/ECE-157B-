import numpy as np
from tensorflow.keras.utils import to_categorical

def load_images(filename):
    return np.load(filename).reshape(-1,64,64,1)

def load_unprocessed_labels(filename):
    return np.load(filename)

label_to_number = {"pass": 0, "total_loss":1, "deform":2, "nodule":3, "edge":4,"crack":5}
number_to_label = {value:key for key, value in label_to_number.items()}
#translate into numbers then make into one hot vectors which we will use keras for 

def process_label(labels):
    # want a python dictionary that translates from string to number 
    #for every label in labls run the label to number dictionary on it 
    number_labels = [label_to_number[label] for label in labels]

    return to_categorical(number_labels, num_classes = 6)

def unprocess_labels(number_labels):
    return [ number_to_label[num]for num in number_labels ]

def load_labels(filename):
    return process_label(load_unprocessed_labels(filename))



