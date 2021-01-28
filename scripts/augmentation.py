
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers 
from image_viewer import show_images 

def augment(X,y):
    print("here")
    #Use data augmentation only on the training set 
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(1, interpolation="nearest", fill_mode= "nearest"), 
    ])

    #augment images and add to list 
    augmented_images = []
    augmented_labels = []
    # stores the number of images for each label 

    for _ in range(1):
        for image, label in zip(X,y):

                #check if the number of image in this label class has exceeded the number of images 
            image = tf.expand_dims(image, 0)
            augmented_image = data_augmentation(image)
            augmented_images.append(augmented_image)
            augmented_labels.append(label)
    
    augmented_images = np.concatenate(augmented_image, axis= 1)
    augmented_labels = np.array(augmented_labels)
    print(augmented_images.shape)

    #manually verify the image quality 
    show_images(augmented_images, augmented_labels)

    np.save("/Users/andreaanez/Downloads/W21/ECE157B/ECE157B_272B_Homework_1/data/augmented_data.npy", augmented_images)
    np.save("/Users/andreaanez/Downloads/W21/ECE157B/ECE157B_272B_Homework_1/data/augmented_labels.npy", augmented_labels)

if __name__== "__main__":
    X = np.load("data/train_data.npy").reshape(-1, 64, 64, 1)
    X = X.astype(np.uint8)
    y = np.load("data/train_label.npy")
    augment(X,y)






# import numpy as np 
# import preprocess as pre 
# import tensorflow.keras.models as models 
# import tensorflow.keras.layers as layers 
# import image_viewer 
# import preprocess as pre 

# augment = models.Sequential(name= "Augment", layers=[
#     layers.experimental.preprocessing.RandomFlip(seed = 42),
#     layers.experimental.preprocessing.RandomRotation(1,fill_mode='nearest'),

# ])

# if __name__ == "__main__":
#     X = pre.load_images("data/train_data.npy").astype(np.float32)
#     Y= pre.load_unprocessed_lables("data/train_label.npy")
#     image_viewer.show_images(augment(X,Y))