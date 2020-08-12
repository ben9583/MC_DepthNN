import glob
import tensorflow as tf
import numpy as np
import os
import sys

FILE = "test.png"
image_height = 512
image_width = 512
input_shape = (image_height, image_width, 3,)
biginator = tf.keras.layers.experimental.preprocessing.Rescaling(255.0)
smallinator = tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)
reshape_layer = tf.keras.layers.Reshape((1, 512, 512, 3))

def decode_colored_img(img):
	img = tf.image.decode_png(img, channels=3)
	return tf.image.resize(img, [image_height, image_width])

def process_path(file_path):
	input_image = tf.io.read_file(file_path)
	input_image = decode_colored_img(input_image)
	print(input_image.shape)
	input_image = smallinator(input_image)

	return input_image#reshape_layer(input_image)

model = tf.keras.models.load_model("model.tf", compile=False)
model.compile(loss="mse", optimizer="adadelta")

data = tf.data.Dataset.from_tensors([process_path(FILE)])
#print(data.shape)
out = model.predict(data, batch_size=1)
out = biginator(out)
out = tf.cast(out[0], tf.uint8)

tf.io.write_file("out.png", tf.image.encode_png(out))