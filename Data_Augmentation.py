from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import numpy as np
import os

datagen = ImageDataGenerator(
        channel_shift_range=10.0,
        vertical_flip=True,
        horizontal_flip=True,
            fill_mode='nearest')

path = '/home/ugupta/Downloads/skin_lesion/data_augmented/valid/malignant/'
for filename in os.listdir(path):
    x = load_img(os.path.join(path + filename))  # this is a PIL image
    img = np.expand_dims(x, axis=0)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0

    for batch in datagen.flow(img, batch_size=1,
                              save_to_dir=path, save_prefix=filename + "_", save_format='jpeg'):
        i += 1
        if i == 2:
            break  # otherwise the generator would loop indefinitely
