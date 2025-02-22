from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

file_name = "data/test/AF/AF07AFFL.JPG.JPG"

img = image.load_img(file_name, target_size=(224,224))

img_array = image.img_to_array(img)

image.save_img('test.jpg', img_array)

img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

saved_model = load_model("vgg16_1.h5")

output = saved_model.predict(img)
print(output)