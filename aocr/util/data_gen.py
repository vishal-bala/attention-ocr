from __future__ import absolute_import

import sys
from warnings import warn

import numpy as np
import tensorflow as tf

from PIL import Image
from six import BytesIO as IO

from .bucketdata import BucketData

try:
    TFRecordDataset = tf.data.TFRecordDataset  # pylint: disable=invalid-name
except AttributeError:
    TFRecordDataset = tf.contrib.data.TFRecordDataset  # pylint: disable=invalid-name


class DataGen(object):
    GO_ID = 1
    EOS_ID = 2
    IMAGE_HEIGHT = 32
    CHARMAP = ['', '', ''] + list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    @staticmethod
    def set_full_ascii_charmap():
        DataGen.CHARMAP = ['', '', ''] + [chr(i) for i in range(32, 127)]
        DataGen.CHARMAP += ['ä', 'ü', 'ö', 'Ä', 'Ü', 'Ö', 'ß',
                            'è', 'È', 'é', 'É', 'á', 'Á', 'à',
                            'À', 'ó', 'Ó', 'ò', 'Ò', 'ï', 'Ï',
                            'â', 'Â', 'ê', 'Ê', 'ô', 'Ô', 'û',
                            'Û', '€', '£', 'ë']

        # We explicitly remove characters that never show up in our training data (so far)
        for char in (
                'Á', '%', '{', 'Ó', 'Ê', 'Ï', 'á', 'â', '=', '^', 'À', 'ï', 'à', 'è', '<', '"', 'é', 'È', '#', 'ä',
                '~', 'Ô', 'ó', '`', '[', 'Ü', '}', 'Â', 'ô', 'Û', ']', 'É', '\\', 'Ö', '>', 'û', '*', 'Ò',
                'ê', '|', 'ò'
        ):
            DataGen.CHARMAP.remove(char)

    def __init__(self,
                 annotation_fn,
                 buckets,
                 epochs=1000,
                 max_width=None):
        """
        :param annotation_fn:
        :param lexicon_fn:
        :param valid_target_len:
        :param img_width_range: only needed for training set
        :param word_len:
        :param epochs:
        :return:
        """
        self.epochs = epochs
        self.max_width = max_width

        self.bucket_specs = buckets
        self.bucket_data = BucketData()

        dataset = TFRecordDataset([annotation_fn])
        dataset = dataset.map(self._parse_record)
        dataset = dataset.shuffle(buffer_size=10000)
        self.dataset = dataset.repeat(self.epochs)

    def clear(self):
        self.bucket_data = BucketData()

    def gen(self, batch_size):

        dataset = self.dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

        images, labels, comments = iterator.get_next()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            while True:
                try:
                    raw_images, raw_labels, raw_comments = sess.run([images, labels, comments])
                    for img, lex, comment in zip(raw_images, raw_labels, raw_comments):

                        if self.max_width and (Image.open(IO(img)).size[0] <= self.max_width):

                            try:
                                word = self.convert_lex(lex)
                            except IndexError as e:
                                raise ValueError("Failed to convert lexicon for {!r}".format(comment)) from e

                            bucket_size = self.bucket_data.append(img, word, lex, comment)
                            if bucket_size >= batch_size:
                                bucket = self.bucket_data.flush_out(
                                    self.bucket_specs,
                                    go_shift=1)
                                yield bucket

                        else:
                            warn("Ignoring {!r} due to size".format(comment))

                except tf.errors.OutOfRangeError:
                    break

        self.clear()

    def convert_lex(self, lex):
        if sys.version_info >= (3,):
            lex = lex.decode('UTF-8') #lex.decode('iso-8859-1')

        assert len(lex) < self.bucket_specs[-1][1]

        return np.array(
            [self.GO_ID] + [self.CHARMAP.index(char) for char in lex] + [self.EOS_ID],
            dtype=np.int32
        )

    @staticmethod
    def _parse_record(example_proto):
        features = tf.parse_single_example(
            example_proto,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'comment': tf.FixedLenFeature([], tf.string, default_value=''),
            })
        return features['image'], features['label'], features['comment']
