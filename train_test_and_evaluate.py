import logging
import pickle
from datetime import datetime

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import (ShuffleSplit, cross_val_score,
                                     train_test_split)

from config import Config


def main():
    logging.basicConfig(
        format='%(asctime)s - %(module)s - %(levelname)-6s - %(message)s',
        level=logging.INFO)
    logging.info(
        'start to extract features from images by using pretrained model')
    start = datetime.utcnow()

    # import features and labels
    with h5py.File(Config.features_path, 'r') as h5f_data:
        features_string = h5f_data['dataset_1']
        features = np.array(features_string)

    with h5py.File(Config.labels_path, 'r') as h5f_label:
        labels_string = h5f_label['dataset_1']
        labels = np.array(labels_string)

    # verify the shape of features and labels
    logging.info('features shape: %r', features.shape)
    logging.info('labels shape: %r', labels.shape)

    logging.info('training started...')
    # split the training and testing data
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        np.array(features),
        np.array(labels),
        test_size=Config.test_size,
        random_state=Config.seed)

    logging.info('splitted train and test data...')
    logging.info('train data  : %r', trainData.shape)
    logging.info('test data   : %r', testData.shape)
    logging.info('train labels: %r', trainLabels.shape)
    logging.info('test labels : %r', testLabels.shape)

    # use logistic regression as the model
    logging.info('creating model...')
    model = LogisticRegression(random_state=Config.seed)
    logging.info('training model...')
    model.fit(trainData, trainLabels)

    # dump classifier to file
    logging.info('saving model...')
    pickle.dump(model, open(Config.classifier_path, 'wb'))

    if Config.cumpute_cv:
        logging.info('cumputing cross-validation\'s scores...')
        cv = ShuffleSplit(
            n_splits=Config.n_splits,
            test_size=Config.test_size,
            random_state=Config.seed)

        scores = cross_val_score(
            model, trainData, trainLabels, cv=cv, n_jobs=-1)

    # use rank-1 and rank-5 predictions
    logging.info('evaluating model...')
    with Config.summary_path.open('w', encoding='utf-8') as f:
        # verify the shape of features and labels
        f.write('features shape: {!r}\n'.format(features.shape))
        f.write('labels shape: {!r}\n\n'.format(labels.shape))
        rank_1 = 0
        rank_5 = 0

        # loop over test data
        for (label, features) in zip(testLabels, testData):
            # predict the probability of each class label and
            # take the top-5 class labels
            predictions = model.predict_proba(np.atleast_2d(features))[0]
            predictions = np.argsort(predictions)[::-1][:5]

            # rank-1 prediction increment
            if label == predictions[0]:
                rank_1 += 1

            # rank-5 prediction increment
            if label in predictions:
                rank_5 += 1

        # convert accuracies to percentages
        rank_1 = (rank_1 / len(testLabels)) * 100
        rank_5 = (rank_5 / len(testLabels)) * 100

        # write the accuracies to file
        f.write('Rank-1: {:.2f}%\n'.format(rank_1))
        f.write('Rank-5: {:.2f}%\n\n'.format(rank_5))
        if Config.cumpute_cv:
            f.write('CrossValidation: \n')
            f.write('Accuracy: ({:.2f} +/- {:.2f})%\n\n'.format(
                scores.mean() * 100,
                scores.std() * 2 * 100))

        # evaluate the model of test data
        preds = model.predict(testData)

        # write the classification report to file
        f.write(classification_report(testLabels, preds))
        f.write('\n')

    if Config.plot_confusion_matrix:
        # display the confusion matrix
        logging.info('confusion matrix')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        # plot the confusion matrix
        cm = confusion_matrix(testLabels, preds)
        sns.heatmap(cm, annot=True, cmap='Set2')
        plt.show()

    end = datetime.utcnow()
    delta = end - start
    seconds = delta.seconds
    hours = seconds / 60 / 60
    minutes = seconds % (60 * 60) / 60
    seconds = seconds % 60
    logging.info('spent %dh %dm %ds', delta.days * 24 + hours, minutes,
                 seconds)


if __name__ == '__main__':
    main()
