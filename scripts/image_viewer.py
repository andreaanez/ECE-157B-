import numpy as np
from matplotlib import pyplot as plt
import preprocess as pre


def show_images(X, y= None):
    plt.figure(figsize=(8,8))
    for i in range(len(X)):
        plt.subplot(5,5,(i%25)+1)
        plt.imshow(X[i])
        plt.clim(0,2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if (y is not None):
            plt.xlabel(y[i])
        
        if((i%25 == 24 or i == len(X)-1) and i>0):
            plt.show(block = False)
            plt.waitforbuttonpress()
            plt.clf()

        
    plt.close()






# def show_image(X,Y):

#     plt.imshow(X)
#     plt.title = Y
#     plt.show() 

# def show_images(X,Y= None):
#     offset = 0
#     _, axs = plt.subplots(3,1,figsize=(8,8))
#     # to get these axis one at a time 
#     for i, ax in enumerate(axs.flatten()):
#         if i+offset >= len(X):
#             break
#         ax.imshow(X[i+offset])
#         ax.set_xticks([])
#         ax.set_yticks([])
#         #ax.set_title(Y[i])
#         if Y is not None:
#             labels = Y
#             ax.set_title(labels[i+offset])

#     plt.show() 
# #only load the data if the main mondule

if __name__ == "__main__":
    X = pre.load_images("/Users/andreaanez/Downloads/W21/ECE157B/ECE157B_272B_Homework_1/data/augmented_data.npy")
    Y = pre.load_unprocessed_labels('/Users/andreaanez/Downloads/W21/ECE157B/ECE157B_272B_Homework_1/data/augmented_labels.npy')
    show_images(X,Y)