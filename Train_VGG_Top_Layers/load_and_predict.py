from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import model_from_json
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Sequential
import keras as keras
from keras import backend as K
print keras.__version__
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import load_model

json_file = open("model_dl4j2.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model_new = model_from_json(loaded_model_json)

#print model_new.summary()
model_new.load_weights('saved_weights_dl4j2.h5')

img_path = '/home/ugupta/Downloads/skin_lesion/ISIC_0000078.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = x.reshape(3,224,224)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model_new.predict_proba(x)
print features

