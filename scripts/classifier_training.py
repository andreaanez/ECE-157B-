import numpy as np
import preprocess as pre
import matplotlib.pyplot as plt 

import tensorflow.keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.callbacks import ModelCheckpoint
# use conv2D for images 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D

from sklearn.model_selection import train_test_split

import tensorflow.keras.layers as layers

import tensorflow.keras.BatchNormalization as BatchNormalization





class WaferCNN:
    #self is the name of the first argument inside a class 
    def __init__(self):
        self.model = None
        self.build_model()



    # needs to be insdie the class function 
    def build_model(self):
        self.model = Sequential(name = "WaferCNN", layers =[ 
            ## flipping exprimental preprocessing function
            # what does the filter do ? 
                # for each kernel get 62X62 image 
            # turned off automatically while testing
            # define input shapew here 

            #generates it dynamically each epcoh -- more epochs might be better 
            layers.InputLayer(input_shape = (64,64,1)),
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            # interpolation at "nearest" and fillmode nearest allows the square image to not look weird when rotated say 15 degrees
            layers.experimental.preprocessing.RandomRotation(1, interpolation= "nearest", fill_mode= "nearest"),

            ## CONVENTIONS ###
            

            Conv2D(16,kernel_size = 4, activation = 'relu'),

            #each time we use the kernel we want to reduce the size of our image 
            # reduces our diemnsionality (determined by the number of paramters) "Curse of Dinemsioanlity"
            # pool size (2x2) means we will take a 4 pixel kernel and condense to one pixel 
            # grabbing pixel of the highest value
            MaxPooling2D((2,2)),


            # automatically turns off a nueron 
            # can it still do feature prediction with out this nueron ?
            # if yes lower the weight 
            # we want to keep the weight values to be as small as possible if not zero 
            # can do our predictions with a small subset of featrues 
            # we want a SPARSE nueral network
            # For example we can ignore the background
            # add after every single feature layer 
            # if training on data thats unimportant this is where it will over fit and predict on eronious features  
            BatchNormalization(),

            # whenever we add a convolution layer we should always increase the number of filters
            # first layer esctracts primiative information 
            # second layer extracts further information off the first layer and so on
            # combianations at each layer level -- can get verry large 
            #set feaure levels to prevent an explosion 
            # 256 filter of information enough to do our classification 
            # the convention is to double the number of filter every time 
            # FILTER SIZE FIRST member in funtion 

            ### ONE REPIITION####
            Conv2D(32, kernel_size = 3, activation = 'relu'),

            #AveragePooling2D(pool_size=(2,2)) <-- muddy information b/c in between values 

            # compresses network
            MaxPooling2D((2,2)),

            Dropout(0.1),
            ### ONE REPIITION####

            #kernel adjsuted by back propigation 
            #begins as noise (random values) learns what is important through each epoch
            Conv2D(64, kernel_size = 3, activation = 'relu'),
            MaxPooling2D((2,2)),
           
            Dropout(0.1),

            Conv2D(128, kernel_size = 3, activation = 'relu'),
            #MaxPooling2D((2,2)), -- can or can't depends.
            

            Flatten(), 

            #How many dense layers we should use 
            # do we need another dense layer between flatten and output 
            # if model complexity not high enough ?
            # add another dense layer (we need allot of data for this will increase complexity allot)
            # if not keep it as it is 

            Dense(6, activation = "softmax")])

            #compress into one hidden layer with 64 nuerons
            # Relu means if its below zero I set it to zero if its above zero I set it to whatever it is 
            # Softmax says if your far below zero then your just zero if your far abov e zero then your one, if your around zero your .5
            #Dense(64,activation = 'relu'),
            #need softmax works well with catagorical cross entropy rates network on amount of entropy of networks gueses trys to reduce entropy, works best with one hot encoding  
          
    


        self.model.compile(optimizer = "adam",
            loss = tensorflow.keras.losses.CategoricalCrossentropy(from_logits = False),
            metrics = ["accuracy"])

        #prints nice summary 
        self.model.summary()


    def train(self, X, y, epochs = 5, batch_size = 1, validation_size =  .20 ):
        #split training data
        x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = validation_size,random_state = 54)

        # if a model is doing really well we want to preserve that model we don't want to loose track of it 
        # so we use:
        filepath = 'best_module.h5'
        checkpoint = ModelCheckpoint(filepath, monitor = "val_accuracy", verbose = 1, save_best_only = True, mode = 'max')
        callbacks_list = [checkpoint]

        # do the train 
        history = self.model.fit(x_train, y_train,
            batch_size = batch_size, epochs = epochs, 
            validation_data= (x_test, y_test),
            callbacks = callbacks_list)

        print(history.history)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.title('Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5,1])
        plt.legend(loc ='lower right')
        plt.show()

        
if __name__== "__main__":
    cnn = WaferCNN()
    # 15:00
    # pass in images here for more variation 
    X = pre.load_images("data/train_data.npy")
    Y = pre.load_labels("data/train_label.npy")
    # 20 epochs 
    # batch size convention is 16 
    # batch of one will just cuase ocillations 
    cnn.train(X,Y, 20, 16)

