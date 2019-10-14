import logging
import pickle

import cv2
import numpy as np
import tensorflow as tf
from keras_preprocessing.image.image_data_generator import ImageDataGenerator
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session

from config import DUMP_DIR

logger = logging.getLogger(__file__)

sess = tf.Session()


class Meta(type):

    __labels_path: str = 'generator_labels.dump'
    __model_name = 'eights-improvement-17-0.75.hdf5'

    def __init__(self, name, bases, attrs):
        logger.info('initialize labels')
        with open(DUMP_DIR / self.__labels_path, 'rb') as file:
            self.LABELS = pickle.load(file)

        # some magic again
        for k, v in self.LABELS.items():
            self.LABELS[k] = '_'.join(self.LABELS[k].split('-')[1:])

        set_session(sess)
        self.MODEL = load_model(DUMP_DIR / self.__model_name)
        self.MODEL._make_predict_function()
        self.graph = tf.get_default_graph()
        super().__init__(name, bases, attrs)


class DogTypesModel(metaclass=Meta):
    # preloaded labels
    LABELS: dict = {}
    # model
    MODEL = None
    WH = (250, 250)

    def __init__(self, *args, **kwargs):
        logger.info('init image_generator')
        super().__init__(*args, **kwargs)
        self.__image_generator = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=30,
            horizontal_flip=True,
            fill_mode='nearest',
            rescale=1.0 / 255.0,
        )

    def predict_function(self, img):
        # some magic
        img = cv2.resize(img, self.WH)
        x = np.expand_dims(np.array(img), 0)
        x = self.__image_generator.apply_transform(x, {})
        self.__image_generator.fit(x)

        with self.graph.as_default():
            steps = 1
            test_generator = self.__image_generator.flow(x,
                                                         shuffle=False)
            set_session(sess)
            results = self.MODEL.predict_generator(test_generator, steps=1)

        r_results = []

        for batch in range(int(len(results) / steps)):
            batch_results = results[batch * steps:(batch + 1) * steps]
            r_results.append(np.average(batch_results, axis=0))

        labels = []
        for result in r_results:
            #  places indices
            rargsort = result.argsort()[::-1][:5]
            for indice in rargsort:
                labels.append(f'{self.LABELS[indice]}: {result[indice]}')
        return labels

    def predict(self, img):
        """ Return predicted img labels with probability
        """
        return self.predict_function(img)
