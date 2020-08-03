import glob
import tensorflow as tf
import numpy as np
import os
import sys

FILE = "data/validation_inputs/4404.png"

image_height = 512
image_width = 512
input_shape = (image_height, image_width, 3,)
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(255.0)

input_layer = tf.keras.Input(shape=(input_shape))
x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu')(input_layer)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(x)      
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)       
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2DTranspose(16, (3, 3), padding="valid", activation='relu')(x)

da_list = []

for i in range(32):
    print(i)
    for j in range(32):
        y = tf.keras.layers.Lambda(lambda x: x[:,i,j,:])(x)
        da_list.append(tf.keras.layers.Dense(1, activation="relu")(y))

x = tf.keras.layers.Concatenate()(da_list)
x = tf.keras.layers.Reshape((32, 32, 1))(x)
x = tf.keras.layers.UpSampling2D(size=(16, 16))(x)
x = tf.keras.layers.Concatenate()([x, input_layer])
output_layer = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer).load_weights("content/training/cp-0200.ckpt")
model.compile(loss="mse", optimizer="adam")

data = tf.io.decode_png(tf.io.read_file(FILE), channels=3)

out = normalization_layer(model.predict(data))

tf.write_file("out.png", tf.io.encode_png(out, channels=1))