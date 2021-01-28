import tensorflow as tf 
import preprocess as pre
import numpy as np

from tensorflow.keras.models import load_model

import tensorflow.keras.utils
import tensorflow.keras.models
from sklearn.metrics import classification_report

# grabing the best model
model = load_model("/Users/andreaanez/Downloads/W21/ECE157B/ECE157B_272B_Homework_1/best_module.h5")

x_test = pre.load_images("data/unknowns.npy")
x_train = pre.load_images("data/train_data.npy")
y_train = pre.load_unprocessed_labels("data/train_label.npy")
tensorflow.keras.utils.plot_model(model, show_shapes = True )



predicted_likelyhoods = model.predict(x_test)
prediction = np.argmax(predicted_likelyhoods, axis=1)
predictions = pre.unprocess_labels(prediction)

y_pred = model.predict(x_train)
y_pred = np.argmax(y_pred, axis=1)
y_pred = pre.unprocess_labels(y_pred)

target_names = ["pass", "total_loss", "deform", "nodule", "edge", "crack"]
print(classification_report(y_train, y_pred, target_names=target_names))

#print(predictions)
# put into a CSV one after another with spces in between 

np.savetxt('prediction.csv', predictions, fmt = '%s')





# model to predict on unknown data 

