import logging
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from sklearn.linear_model import LogisticRegression  # noqa

from config import Config
from model import GetModel

logging.basicConfig(
    format='%(asctime)s - %(module)s - %(levelname)-6s - %(message)s',
    level=logging.INFO)
logging.info('start to extract features from images by using pretrained model')

# load the trained logistic regression classifier
logging.info('loading the classifier...')
classifier = pickle.load(Config.classifier_path.open('rb'))

# pretrained models needed to perform feature extraction on test data too!
model, image_size, preprocess_input = GetModel(Config.model_name)
logging.info('successfully loaded the pretrained model %r ...',
             Config.model_name)

# get all the train labels
class_pathes = {
    class_path.name: class_path
    for class_path in Config.dataset_path.glob('*')
}
train_labels = sorted(class_pathes.keys())

# get all the target images paths
target_images = Config.predict_path.glob('*.jpg')

# loop through each image in the target data
for image_path in target_images:
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
    # expand the dimensions, make 3D to 4D.
    flat = np.expand_dims(flat, axis=0)
    # perform prediction on test image
    preds = classifier.predict(flat)
    prediction = train_labels[preds[0]]

    logging.info('I think it is a %s, image \'%s\'', prediction, image_path)
    # cv2.imread only accept str instance
    traget_img = mpimg.imread(image_path)
    predict_imgs = class_pathes[prediction].glob('*.jpg')
    predict_img_path = next(predict_imgs)
    predict_img = mpimg.imread(predict_img_path)
    ax = plt.subplot(121)
    ax.set_title('target')
    ax.imshow(traget_img)
    ax = plt.subplot(122)
    ax.imshow(predict_img)
    ax.set_title('predict %s' % prediction)
    plt.show()
