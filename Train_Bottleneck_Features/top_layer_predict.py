from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import model_from_json

model = VGG16(weights='imagenet', include_top=False)

img_path = '/home/ugupta/Downloads/skin_lesion/ISIC_0000078.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

json_file = open("vggmodel_top.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
latest_weights_filename = 'bottleneck_fc_model.h5'
loaded_model.load_weights(latest_weights_filename)

loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
probs = loaded_model.predict(features, batch_size=8, verbose=1)
print np.round(probs)