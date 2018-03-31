from datetime import datetime
import logging

import h5py
import numpy as np
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

from model import GetModel
from config import Config

logging.basicConfig(
    format='%(asctime)s - %(module)s - %(levelname)-6s - %(message)s',
    level=logging.INFO)
logging.info('start to extract features from images by using pretrained model')

start = datetime.utcnow()

# create the pretrained models
model, image_size, preprocess_input = GetModel(Config.model_name)

logging.info('successfully loaded the pretrained model %r ...',
             Config.model_name)

# get all the dataset labels
class_pathes = {
    class_path.name: class_path
    for class_path in Config.dataset_path.glob('*')
}
train_labels = sorted(class_pathes.keys())

# variables to hold features and labels
features = []
labels = []

# loop over all the labels in the folder
logging.info('processing extracting features...')
for label_no, label in enumerate(train_labels, start=1):
    images_path = class_pathes[label]
    for image_no, image_path in enumerate(images_path.glob('*.jpg'), start=1):
        # load image and resize with image size
        # that fits the model's input layer.
        img = image.load_img(image_path, target_size=image_size)
        # convert PIL Image instance into np.array.
        x = image.img_to_array(img)
        # expand the dimensions, make 3D to 4D.
        x = np.expand_dims(x, axis=0)
        # preprocessing the data by model's special preprocess function.
        x = preprocess_input(x)
        # extracting the features.
        feature = model.predict(x)
        flat = feature.flatten()

        # collecting image's features and label
        features.append(flat)
        labels.append(label)

        logging.debug('processed %d images', image_no)
    logging.info('completed no.%d label %r', label_no, label)

# list[np.array] => np.array
features = np.array(features)

logging.info('features\' shape: %r', features.shape)

# save features and labels
with h5py.File(Config.features_path, 'w') as h5f_data:
    h5f_data.create_dataset('dataset_1', data=features)
logging.info('features saved...')

# encode the labels using LabelEncoder
logging.info('encoding labels...')
le = LabelEncoder()
le.fit(train_labels)
le_labels = le.fit_transform(labels)

with h5py.File(Config.labels_path, 'w') as h5f_label:
    h5f_label.create_dataset('dataset_1', data=np.array(le_labels))
logging.info('labels saved...')

# # save model
# model_json = model.to_json()
# output_path = Config.model_path.with_suffix('.json')
# with output_path.open('w', encoding='utf-8') as json_file:
#     json_file.write(model_json)
# logging.info('model saved...')

# # save weights
# output_path = Config.model_path.with_suffix('.h5')
# model.save_weights(output_path)
# logging.info('model layers\' weights saved...')

# totaling spent time
end = datetime.utcnow()
delta = end - start
seconds = delta.seconds
hours = seconds / 60 / 60
minutes = seconds % (60 * 60) / 60
seconds = seconds % 60
logging.info('spent %dh %dm %ds', delta.days * 24 + hours, minutes, seconds)
