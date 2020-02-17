'''
Modelbuilder for regression model

Author: Jagadish Venkataraman
Date: 4/16/2019
'''
import os
import math
import tensorflow as tf


class ModelBuilder(object):
    def __init__(self, config):

        self.mode = config.get('mode', 'train')
        self.IMG_HEIGHT = config.get('IMG_HEIGHT', 256)
        self.IMG_WIDTH = config.get('IMG_WIDTH', 256)
        self.backbone_network = config.get('backbone_network', None)
        self.num_channels = config.get('num_channels', 1)
        self.logdir = config.get('logdir', None)
        self.lr = config.get('lr', 0.001)

        if self.mode == 'train':
            self.create_log_dir_and_callbacks()

        if self.backbone_network is None:
            self.create_conv_net_plus_fcn()
        elif self.backbone_network is 'resnet':
            self.create_resnet_plus_fcn()
        elif self.backbone_network is 'inception':
            self.create_inception_plus_fcn()
        else:
            raise ValueError("Unknown backbone network specified")

        self.model.summary()


    def create_log_dir_and_callbacks(self):
        '''
        Make the needed directories and define the callbacks
        '''
        # make log directory and define callbacks
        os.makedirs(self.logdir)

        # Creating Keras callbacks
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir, histogram_freq=1)
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(self.logdir + '/checkpoints/' +
            'weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5, save_weights_only=True)
        os.makedirs(self.logdir + '/checkpoints', exist_ok=True)
        self.early_stopping_checkpoint = tf.keras.callbacks.EarlyStopping(patience=300)
        self.lrate = tf.keras.callbacks.LearningRateScheduler(self.step_decay)


    def weighted_mse_loss(self, yTrue, yPred):
        '''
        Weighted MSE custom loss function
        '''
        cond_1 = tf.cast(tf.keras.backend.equal(yTrue, 0), tf.int32)
        weights = tf.cast(cond_1*tf.cast(tf.constant(10), tf.int32) + (1 - cond_1)*tf.cast((yTrue + 3), tf.int32), tf.float32)
        return tf.keras.backend.mean(weights*tf.keras.backend.square(yTrue-yPred))

        # return tf.keras.backend.mean(tf.keras.backend.minimum(yTrue+3, 15)*tf.keras.backend.square(yTrue-yPred))


    def create_conv_net_plus_fcn(self):
        '''
        Method to build a conv-net backbone network + FCN for regression
        '''
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', data_format='channels_last',
                        input_shape=(self.IMG_HEIGHT,self.IMG_WIDTH,self.num_channels)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Activation('relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='valid'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Activation('relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='valid'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Activation('relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='valid'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Activation('relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        # self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        # self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(1, activation='linear'))

        self.model.compile(optimizer='adam',
                      loss=self.weighted_mse_loss,
                      metrics=['mae'])


    def create_inception_plus_fcn(self):
        '''
        Method to create IV3 backbone + FCN with regression head
        '''
        base_model = tf.keras.applications.InceptionV3(input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH,3),
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False
        # Let's take a look at the base model architecture
        base_model.summary()

        self.model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512,
                              activation='relu'),
            tf.keras.layers.Dense(128,
                              activation='relu'),
            tf.keras.layers.Dense(1,
                               activation='linear')
        ])

        self.model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mae'])


    def create_resnet_plus_fcn(self):
        '''
        Method to create resnet backbone + FCN with regression head
        '''
        base_model = tf.keras.applications.ResNet50(input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH,3),
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False
        # Let's take a look at the base model architecture
        base_model.summary()

        self.model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512,
                              activation='relu'),
            tf.keras.layers.Dense(128,
                              activation='relu'),
            tf.keras.layers.Dense(1,
                               activation='linear')
        ])

        self.model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mae'])


    def step_decay(self, epoch):
       initial_lrate = self.lr
       drop = 1
       epochs_drop = 20.0
       lrate = initial_lrate * math.pow(drop,
               math.floor((1+epoch)/epochs_drop))
       return lrate
