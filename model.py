def GetModel(model_name):
    """
    according the model name return the specific pretrained model.

    Parameters
    ----------
    model_name : str
        pretrained model name,
        inceptionv3, resnet50, vgg16, vgg19 and xception.

    Returns
    -------
    model : keras.Model
        a pretrained model
    image_size : tuple
        image_size for model input layer, models' default config.
    preprocess_input : Function
        preprocess image data for model.
    """
    if model_name == 'VGG16':
        from keras.applications.vgg16 import VGG16, preprocess_input
        model = VGG16(include_top=False, weights='imagenet', pooling='avg')
        image_size = (224, 224)
    elif model_name == 'VGG19':
        from keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(include_top=False, weights='imagenet', pooling='avg')
        image_size = (224, 224)
    elif model_name == 'ResNet50':
        from keras.applications.resnet50 import ResNet50, preprocess_input
        model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
        image_size = (224, 224)
    elif model_name == 'InceptionV3':
        from keras.applications.inception_v3 import (InceptionV3,
                                                     preprocess_input)
        model = InceptionV3(
            include_top=False, weights='imagenet', pooling='avg')
        image_size = (299, 299)
    elif model_name == 'InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import (InceptionResNetV2,
                                                            preprocess_input)
        model = InceptionResNetV2(
            include_top=False, weights='imagenet', pooling='avg')
        image_size = (299, 299)
    elif model_name == 'MobileNet':
        from keras.applications.mobilenet import MobileNet, preprocess_input
        model = MobileNet(
            include_top=False,
            weights='imagenet',
            pooling='avg',
            input_shape=(224, 224, 3))
        image_size = (224, 224)
    elif model_name == 'Xception':
        from keras.applications.xception import Xception, preprocess_input
        model = Xception(include_top=False, weights='imagenet', pooling='avg')
        image_size = (299, 299)
    elif model_name == 'DenseNet121':
        from keras.applications.densenet import DenseNet121, preprocess_input
        model = DenseNet121(
            include_top=False, weights='imagenet', pooling='avg')
        image_size = (224, 224)
    elif model_name == 'DenseNet169':
        from keras.applications.densenet import DenseNet169, preprocess_input
        model = DenseNet169(
            include_top=False, weights='imagenet', pooling='avg')
        image_size = (224, 224)
    elif model_name == 'DenseNet201':
        from keras.applications.densenet import DenseNet201, preprocess_input
        model = DenseNet201(
            include_top=False, weights='imagenet', pooling='avg')
        image_size = (224, 224)
    elif model_name == 'NASNetLarge':
        from keras.applications.nasnet import NASNetLarge, preprocess_input
        model = NASNetLarge(
            include_top=False, weights='imagenet', pooling='avg')
        image_size = (331, 331)
    elif model_name == 'NASNetMobile':
        from keras.applications.nasnet import NASNetMobile, preprocess_input
        model = NASNetMobile(
            include_top=False, weights='imagenet', pooling='avg')
        image_size = (224, 224)
    else:
        raise ValueError('invalid model name %r.' % model_name)

    return model, image_size, preprocess_input
