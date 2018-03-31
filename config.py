from pathlib import Path


class Config:
    model_name = 'DenseNet169'
    # VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2,
    # Xception, MobileNet, DenseNet121, DenseNet169, DenseNet201,
    # NASNetLarge, NASNetMobile
    dataset_name = '17flowers'
    # 17flowers, 102flowers

    dataset_path = Path('dataset') / dataset_name / 'organized'
    predict_path = Path('dataset/predict_targets')
    out_path = Path('output') / dataset_name / model_name
    features_path = out_path / 'features.h5'
    labels_path = out_path / 'labels.h5'
    summary_path = out_path / 'summary.txt'
    model_path = out_path / 'model'
    classifier_path = out_path / 'classifier.pickle'

    # test dataset size.
    test_size = 0.10
    # n_splits,
    # Number of re-shuffling & splitting iterations for Cross-Validator.
    n_splits = 10
    # random seed, using for spilt the dataset into train and test dataset.
    # set one specific value to keep the same result in any situations.
    seed = 0

    plot_confusion_matrix = False
    cumpute_cv = True


if not Config.predict_path.exists():
    Config.predict_path.mkdir(parents=True)

if not Config.out_path.exists():
    Config.out_path.mkdir(parents=True)
