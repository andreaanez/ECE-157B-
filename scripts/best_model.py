import tensorflow as tf 
import preprocess as pre
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.utils
import tensorflow.keras.models
import tensorflow.math as math 
import image_viewer as img

# grabing the best model
model = load_model("/Users/andreaanez/Downloads/W21/ECE157B/ECE157B_272B_Homework_1/best_module.h5")
X = pre.load_images("data/train_data.npy")
Y = np.argmax(pre.load_labels("data/train_label.npy"), axis=1)

tensorflow.keras.utils.plot_model(model, show_shapes = True )

predicted_likelyhoods = model.predict(X)
predictions = np.argmax(predicted_likelyhoods, axis=1)
#predictions = pre.unprocess_lables(prediction)

#list of images that where misprdicted
img.show_images(X[Y != predictions], predictions[Y != predictions])

matrix = math.confusion_matrix(Y, predictions)
print(matrix)

#print(predictions)
# put into a CSV one after another with spces in between 

