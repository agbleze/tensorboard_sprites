# %%
import os
import glob

import cv2
import numpy as np
import tensorflow as tf
#from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins import projector
# %%
LOG_DIR = "./tensorboard-logs"
IMAGES_DIR = "./XO_images"
IMAGE_SIZE = (64, 64)
SPRITES_FILE = "sprites.png"
SPRITES_PATH = os.path.join(LOG_DIR, SPRITES_FILE)
FEATURE_VECTORS = "feature_vectors.npy"
METADATA_FILE = "metadata.tsv"
METADATA_PATH = os.path.join(LOG_DIR, METADATA_FILE)
CHECKPOINT_FILE = os.path.join(LOG_DIR, "features.ckpt")

# Max sprite size is 8192 x 8192 so this max samples makes visualization easy
MAX_NUMBER_SAMPLES = 8191 

# %%
def create_sprite(data):
    """
    Tile images into sprite image. 
    Add any necessary padding
    """
    
    # For B&W or greyscale images
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0), (0, 0))
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    
    # Tile images into sprite
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    # print(data.shape) => (n, image_height, n, image_width, 3)
    
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # print(data.shape) => (n * image_height, n * image_width, 3) 
    return data

# %%
# Create sprite image

image_files = glob.glob(os.path.join(IMAGES_DIR, "*.png"))

img_data = []
for img in image_files[:MAX_NUMBER_SAMPLES]:
    input_img = cv2.imread(img)
    input_img_resize = cv2.resize(input_img, IMAGE_SIZE) 
    img_data.append(input_img_resize)
img_data = np.array(img_data)

sprite = create_sprite(img_data)
cv2.imwrite(SPRITES_PATH, sprite)

# %%
# Create metadata, configure for tensorboard embedding

# Create metadata
# Can include class data in here if interested / have available
with open(METADATA_PATH, 'w+') as wrf:
    wrf.write("\n".join([str(a) for a,i in enumerate(image_files[:MAX_NUMBER_SAMPLES])]))

feature_vectors = np.load(FEATURE_VECTORS)

features = tf.Variable(feature_vectors[:MAX_NUMBER_SAMPLES], name='features')

# Write summaries for tensorboard
with tf.Session() as sess:
    saver = tf.train.Saver([features])

    sess.run(features.initializer)
    saver.save(sess, CHECKPOINT_FILE)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    embedding.metadata_path = METADATA_FILE

    # This adds the sprite images
    embedding.sprite.image_path = SPRITES_FILE
    embedding.sprite.single_image_dim.extend(IMAGE_SIZE)
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)


