import glob
import tensorflow as tf
import numpy as np
import os
import sys
import wandb
from wandb.keras import WandbCallback

tf.get_logger().setLevel('ERROR')

#os.environ['WANDB_MODE'] = 'dryrun' # Testing for funcitonality
wandb.init(project="mc_depthnn")

physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	# Invalid device or cannot modify virtual devices once initialized
	pass

AUTOTUNE = tf.data.experimental.AUTOTUNE
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "images")
BATCH_SIZE = 8
EPOCHS = 200

image_count = len(glob.glob(DATA_DIR + os.path.sep + "inputs"))
image_width = 512
image_height = 512

input_shape = (image_height, image_width, 3,)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

def decode_colored_img(img):
	img = tf.image.decode_png(img, channels=3)
	return tf.image.resize(img, [image_height, image_width])

def decode_bw_img(img):
	img = tf.image.decode_png(img, channels=1)
	return tf.image.resize(img, [image_height, image_width])

def process_path(file_path):
	input_image = tf.io.read_file(file_path)
	output_image = tf.io.read_file(tf.strings.join([DATA_DIR, "/outputs/", tf.strings.substr(file_path, -8, -1)]))

	input_image = decode_colored_img(input_image)
	output_image = decode_bw_img(output_image)

	input_image = normalization_layer(input_image)
	output_image = normalization_layer(output_image)

	return input_image, output_image

def process_validation_path(file_path):
	input_image = tf.io.read_file(file_path)
	output_image = tf.io.read_file(tf.strings.join([DATA_DIR, "/validation_outputs/", tf.strings.substr(file_path, -8, -1)]))

	input_image = decode_colored_img(input_image)
	output_image = decode_bw_img(output_image)

	input_image = normalization_layer(input_image)
	output_image = normalization_layer(output_image)

	return input_image, output_image

def configure_for_performance(ds):
	#ds = ds.cache()
	ds = ds.batch(BATCH_SIZE)
	ds = ds.prefetch(buffer_size=AUTOTUNE)
	return ds

print("Creating model...")
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(input_shape)))
model.add(tf.keras.layers.MaxPooling2D((3, 3)))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((3, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((3, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((3, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Reshape((16, 16, 4)))
model.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation='relu'))
model.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation='relu'))
model.add(tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, padding="same", activation='relu'))
model.add(tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=2, padding="same", activation='relu'))
model.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=2, padding="same", activation='sigmoid'))
"""

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
x = tf.keras.layers.Conv2DTranspose(1, (1, 1), padding="valid", activation='relu')(x)

"""
da_list = []

for i in range(32):
    print(i)
    for j in range(32):
        y = tf.keras.layers.Lambda(lambda x: x[:,i,j,:])(x)
        da_list.append(tf.keras.layers.Dense(1, activation="relu")(y))

x = tf.keras.layers.Concatenate()(da_list)
x = tf.keras.layers.Reshape((32, 32, 1))(x)
"""

x = tf.keras.layers.UpSampling2D(size=(16, 16))(x)
x = tf.keras.layers.Concatenate()([x, input_layer])
output_layer = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
     
#model.summary()

model.compile(loss='MSE', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=[tf.keras.metrics.Accuracy()])

checkpoint_path = os.path.realpath(__file__)[:len(os.path.realpath(__file__)) - 8] + "/content/training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1,
    period=100
)

model.save_weights(checkpoint_path.format(epoch=0))

print("Compiling data...")

train_ds = tf.data.Dataset.list_files(str(DATA_DIR + '/inputs/*.png'), shuffle=False).take(4096)
train_ds = train_ds.shuffle(4096, reshuffle_each_iteration=True)
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
train_ds = configure_for_performance(train_ds)

test_ds = tf.data.Dataset.list_files(str(DATA_DIR + '/validation_inputs/*.png'), shuffle=False)
test_ds = test_ds.shuffle(4096, reshuffle_each_iteration=True)
test_ds = test_ds.map(process_validation_path, num_parallel_calls=AUTOTUNE)
test_ds = configure_for_performance(test_ds)

#print(type(train_ds))
#print(train_ds)
#print(type(test_ds))
#print(test_ds)

model.fit(train_ds, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, callbacks=[cp_callback, WandbCallback()], validation_data=test_ds)
model.save(os.path.join(wandb.run.dir, "model.h5"))
