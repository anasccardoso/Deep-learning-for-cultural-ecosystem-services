import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import xlrd
import cv2
import os, sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.resnet import ResNet152
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.metrics import specificity_score
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img

#Importing the images
images_names = sorted(os.listdir("/content/"), key = lambda l: (len(l), l))
extension = ".jpg"
images_cv = [cv2.imread(i) for i in images_names if i.endswith(extension)]
x_train = np.array(images_cv)

#Importing the photo labels
df = pd.read_excel("/content/CA_PNPG_Flickr.xlsx")
data = df.dropna(subset = ["Main_focus"])
y = data["Main_focus"]
y_train = np.array(y)

#Normalization and Scaling of the data
x_train = x_train.astype('float32')
x_train /= 255
print('x_train shape:', x_train.shape)

#Parameters
batch_size = 10
num_classes = 2
epochs = 50

#Converting class vectores to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

#Check GPU
print(tf.config.list_physical_devices('GPU'))

#Training with 5-fold-cross validation
seed = 7
np.random.seed(seed)
kfold = KFold(n_splits = 5, shuffle = True, random_state = seed)
cvscores = []
kf = kfold.split(x_train, y_train)
train, test = next(kf) #first fold
train, test = next(kf) #second fold
train, test = next(kf) #third fold
train, test = next(kf) #fourth fold
train, test = next(kf) #fifth fold

datagen = ImageDataGenerator(rotation_range = 0, horizontal_flip = True, width_shift_range = 0.2, height_shift_range = 0.2, zoom_range = 0.2, data_format = "channels_last")
datagen.fit(x_train[train])
imageGen = datagen.flow(x_train[train], y_train[train], batch_size = 1)
x = np.copy(x_train[train])
y = np.copy(y_train[train])
for i in x:
  for j in range(5):
    batch = imageGen.next()
    x = np.append(x, batch[0], axis = 0)
    y = np.append(y, batch[1], axis = 0)
model = Sequential()
model.add(ResNet152(include_top = False, pooling = "max", input_shape = (173, 273, 3), weights = "imagenet")) #for VGG16, use VGG16()
model.add(Dense(128, activation = 'relu'))
model.add(Dense(num_classes, activation = "softmax"))
model.compile(optimizer = keras.optimizers.Adam(lr = 0.0001), loss = keras.losses.categorical_crossentropy, metrics = ['accuracy']) #for VGG16, use lr = 0.000001
es = EarlyStopping(monitor='val_loss', mode = 'auto', patience = 16, verbose = 0)
cp = ModelCheckpoint('/content/model.h5', monitor='val_loss', save_best_only = True, verbose = 0, mode = 'auto')
history = model.fit(x, y, validation_split = 0.1, batch_size = batch_size, epochs = epochs, callbacks = [es, cp], verbose = 0)

#Plot the training parameters
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()

#Evaluate the model
new_model = load_model('/content/model.h5')
scores = new_model.evaluate(x_train[test], y_train[test], verbose = 0)
print("%s: %.2f%%" % (new_model.metrics_names[1], scores[1] * 100))
cvscores.append(scores[1] * 100)
y_pred = new_model.predict_classes(x_train[test], verbose = 0)
rounded_labels = np.argmax(y_train[test], axis = 1)
print('Confusion Matrix')
confusion = confusion_matrix(rounded_labels, y_pred)
print(confusion)
print("Sensitivity:", round(recall_score(rounded_labels, y_pred, average = "macro") * 100, 2))
print("Specificity:", round(specificity_score(rounded_labels, y_pred, average = "macro") * 100, 2))
print("F1 score:", round(f1_score(rounded_labels, y_pred, average = "macro") * 100, 2))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
#print the images and the respective predicted and actual labels
i = 0
while i < 100: 
  plt.imshow(x_train[test][i])
  print("Predicted label: " + str(y_pred[i]))
  print("Actual label: " + str(y_train[test][i]))
  i += 1
  plt.show()
