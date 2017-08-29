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

input_tensor = Input(shape=(3,224,224))
# json_file = open("VGG16NoTop.json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

loaded_model = keras.applications.VGG16(weights='imagenet', 
                           include_top=False,
                           input_tensor=input_tensor)

model_new = Sequential()
for layer in loaded_model.layers:
	layer.trainable = False
 	model_new.add(layer)	

latest_weights_filename = 'vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
model_new.load_weights(latest_weights_filename)

model_new.add(Flatten(input_shape=model_new.output_shape[1:]))
model_new.add(Dense(4096, activation='relu'))
model_new.add(Dropout(0.5))
model_new.add(Dense(4096, activation='relu'))
model_new.add(Dropout(0.5))
model_new.add(Dense(2, activation='sigmoid'))

#train model
DATA_HOME_DIR = '/media/ugupta/data/skin_lesion/skin_lesion_data_split'
test_path = DATA_HOME_DIR + '/test/' #We use all the test data
results_path=DATA_HOME_DIR + '/results/'
train_path=DATA_HOME_DIR + '/train/'
valid_path=DATA_HOME_DIR + '/valid/'

gen=image.ImageDataGenerator(rescale=1./255)
train_batches = gen.flow_from_directory(train_path, target_size=(224,224),
                class_mode='categorical', shuffle=True, batch_size=64)
val_batches = gen.flow_from_directory(valid_path, target_size=(224,224),
                class_mode='categorical', shuffle=True, batch_size=64)

model_new.compile(optimizer='sgd',
                loss='categorical_crossentropy', metrics=['accuracy'])

model_new.fit_generator(train_batches, samples_per_epoch=train_batches.nb_sample, nb_epoch=10,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample)

for layer in model_new.layers:
	layer.trainable = True 	

model_json = model_new.to_json()
with open("model_dl4j2.json", "w") as json_file:
      json_file.write(model_json)

model_new.save_weights('saved_weights_dl4j2.h5')


img_path = '/home/ugupta/Downloads/skin_lesion/ISIC_0000078.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = x.reshape(3,224,224)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model_new.predict_proba(x)
print features


