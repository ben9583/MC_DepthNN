import glob
import tensorflow as tf
import numpy as np
import os
import sys
import wandb
from wandb.keras import WandbCallback

tf.get_logger().setLevel('ERROR')

os.environ['WANDB_MODE'] = 'dryrun' # Testing for funcitonality
wandb.init(project="MC_DepthNN")

physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	# Invalid device or cannot modify virtual devices once initialized
	pass

AUTOTUNE = tf.data.experimental.AUTOTUNE
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "images")
BATCH_SIZE = 8
EPOCHS = 3

image_count = len(glob.glob(DATA_DIR + os.path.sep + "inputs"))
image_width = 512
image_height = 512

input_shape = (image_height, image_width, 3,) # 3 = color channels

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

def configure_for_performance(ds):
	ds = ds.cache()
	ds = ds.shuffle(buffer_size=1000)
	ds = ds.batch(BATCH_SIZE)
	ds = ds.prefetch(buffer_size=AUTOTUNE)
	return ds

#print(normalization_layer(decode_bw_img(tf.io.read_file(os.path.join(DATA_DIR, "outputs", os.path.basename("/Users/ben/Desktop/Git_Stuff/MC_DepthNN_Git/data/images/inputs/0001.png"))))))

print("Creating model...")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(input_shape)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Reshape((16, 16, 4)))
model.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation='relu'))
model.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation='relu'))
model.add(tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, padding="same", activation='relu'))
model.add(tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=2, padding="same", activation='relu'))
model.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=2, padding="same", activation='sigmoid'))

model.summary()
model.compile(loss='MSE', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

checkpoint_path = "/Users/ben/Desktop/Git_Stuff/MC_DepthNN_Git/content/training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1,
    period=200
)

model.save_weights(checkpoint_path.format(epoch=0))

print("Compiling data...")

train_ds = tf.data.Dataset.list_files(str(DATA_DIR + '/inputs/*.png'))
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for val in train_ds.take(1):
	print(val)

#train_ds = train_ds.map(lambda x, y: (x.set_shape([512, 512, 3]), y))
#train_ds = train_ds.map(lambda x, y: (x, y.set_shape([512, 512, 1])))

#print("DID STUFF HERE")
#for val in train_ds.take(1):
#	print(val)

train_ds = configure_for_performance(train_ds)

test_ds = tf.data.Dataset.list_files(str(DATA_DIR + '/validation_inputs/*.png'))
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
#test_ds = test_ds.map(lambda x, y: (x.set_shape([512, 512, 3]), y))
#test_ds = test_ds.map(lambda x, y: (x, y.set_shape((512, 512, 1))))
test_ds = configure_for_performance(test_ds)

print(type(train_ds))
print(train_ds)
print(type(test_ds))
print(test_ds)

model.fit(train_ds, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, callbacks=[cp_callback, WandbCallback()], validation_data=test_ds)
