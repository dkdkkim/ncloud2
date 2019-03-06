import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)

##########################################################################################
#pre processing

train_dir = '/home/dkkim/downloads/tiny-imagenet-200/train'
val_dir = '/home/dkkim/downloads/tiny-imagenet-200/val'

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,width_shift_range =0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,horizontal_flip=True,fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(128,128),batch_size=100,class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(val_dir,target_size=(128,128),batch_size=100,class_mode='categorical')

##########################################################################################
from keras import layers
from keras import models
from keras import regularizers
from keras.layers import Dense, merge, Concatenate, concatenate, Input, Dropout, Conv2D
from keras.layers.normalization import BatchNormalization


def resnet(input, n_ch):
    x1 = layers.convolutional.ZeroPadding2D((1, 1))(input)
    x1 = Conv2D(n_ch, (3, 3), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.convolutional.ZeroPadding2D((1, 1))(x1)
    x1 = Conv2D(n_ch, (3, 3), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    output = concatenate(inputs=[input, x1])
    output = layers.Activation('relu')(output)
    output = Conv2D(n_ch, (1, 1), kernel_initializer='he_normal')(output)
    return output


# def skip_connect(input,n_ch):

#     x1 = Conv2D(n_ch/2,(1,1), kernel_initializer='he_normal')(input)
#     x1 = BatchNormalization()(x1)
#     x1 = layers.Activation('relu')(x1)
#     x1 = layers.convolutional.ZeroPadding2D((1,1))(x1)
#     x1 = Conv2D(n_ch,(3,3), kernel_initializer='he_normal')(x1)
#     x1 = BatchNormalization()(x1)
#     x1 = layers.Activation('relu')(x1)
#     x1 = Conv2D(n_ch/2,(1,1), kernel_initializer='he_normal')(x1)
#     x1 = BatchNormalization()(x1)
#     x1 = layers.Activation('relu')(x1)

#     x2 = Conv2D(n_ch/2,(1,1), kernel_initializer='he_normal')(input)
#     x2 = BatchNormalization()(x2)
#     x2 = layers.Activation('relu')(x2)

#     output = concatenate(inputs = [x2,x1])
#     return output

def skip_connect(input, n_ch):
    x1 = BatchNormalization()(input)
    x1 = layers.Activation('relu')(x1)
    x1 = Conv2D(n_ch / 2, (1, 1), kernel_initializer='he_normal')(x1)
    x1 = layers.convolutional.ZeroPadding2D((1, 1))(x1)
    x1 = BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = Conv2D(n_ch, (3, 3), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = Conv2D(n_ch / 2, (1, 1), kernel_initializer='he_normal')(x1)

    x2 = BatchNormalization()(input)
    x2 = layers.Activation('relu')(x2)
    x2 = Conv2D(n_ch / 2, (1, 1), kernel_initializer='he_normal')(x2)

    output = concatenate(inputs=[x2, x1])
    return output


input_dat = Input(shape=(128, 128, 3))

first_layer = layers.convolutional.ZeroPadding2D((1, 1))(input_dat)
first_layer = BatchNormalization()(first_layer)
first_layer = Conv2D(64, (3, 3), kernel_initializer='he_normal')(first_layer)
# first_layer = layers.Activation('relu')(first_layer)
# first_layer = layers.MaxPooling2D((2,2))(first_layer)

x1 = skip_connect(first_layer, 64)
x1 = skip_connect(x1, 64)
x1 = layers.MaxPooling2D((2, 2))(x1)

x2 = skip_connect(x1, 128)
x2 = skip_connect(x2, 128)
x2 = skip_connect(x2, 128)
x2 = layers.MaxPooling2D((2, 2))(x2)

x2_2 = skip_connect(x2, 256)
x2_2 = skip_connect(x2_2, 256)
x2_2 = skip_connect(x2_2, 256)
x2_2 = layers.MaxPooling2D((2, 2))(x2_2)

x3 = skip_connect(x2_2, 512)
x3 = skip_connect(x3, 512)
x3 = skip_connect(x3, 512)
x3 = layers.MaxPooling2D((2, 2))(x2_2)

x4 = skip_connect(x3, 1024)
x4 = skip_connect(x4, 1024)
x4 = skip_connect(x4, 1024)

output = layers.AveragePooling2D((8, 8))(x4)
output = layers.Flatten()(output)
output = layers.Dense(200, kernel_regularizer=regularizers.l2(0.001), activation='softmax',
                      kernel_initializer='he_normal')(output)
# output = layers.Dense(200,activation='relu', kernel_initializer='he_normal')(output)

model = models.Model(inputs=input_dat, outputs=output)

model.summary()
##########################################################################################
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
import os, shutil
import math


# learning rate log
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


## learning rate decay

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop, math.floor(
        (1 + epoch) / epochs_drop))  # lr = lr0 * drop^floor(epoch / epochs_drop)
    return lrate


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))


loss_history = LossHistory()

## model compile

optimizer = optimizers.RMSprop(lr=0.001)
lr_metric = get_lr_metric(optimizer)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', lr_metric])
lrate = LearningRateScheduler(step_decay)
# check point
log_dir = '/home/dkkim/documents/tiny_3_log'
log_data_dir = os.path.join(log_dir, 'log_7')  ################ version
if not os.path.exists(log_data_dir): os.mkdir(log_data_dir)

filepath = os.path.join(log_data_dir, 'weights.{epoch:02d}-{val_acc:.2f}.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=0, save_best_only=False, save_weights_only=False,
                             mode='auto', period=1)
callbacks_list = [checkpoint, lrate]

# training
history = model.fit_generator(train_generator, steps_per_epoch=1000, epochs=50, callbacks=callbacks_list,
                              validation_data=validation_generator, validation_steps=50)

# model.save

model_weight = os.path.join(log_data_dir, 'tiny_3_imagenet.h5')
model_arch = os.path.join(log_data_dir, 'tiny_3_imagenet.json')

model.save(model_weight)
with open(model_arch, 'w') as f:
    f.write(model.to_json())

import csv

dict = history.history

csv_path = os.path.join(log_data_dir, 'tiny_3_log.csv')

w = csv.writer(open(csv_path, "w"))
for key, val in dict.items():
    w.writerow([key, val])

##########################################################################################

# model.save

model_weight = os.path.join(log_data_dir, 'tiny_2_imagenet.h5')
model_arch = os.path.join(log_data_dir, 'tiny_2_imagenet.json')

model.save(model_weight)
with open(model_arch, 'w') as f:
    f.write(model.to_json())

import csv

dict = history.history

csv_path = os.path.join(log_data_dir, 'tiny_2_log.csv')

w = csv.writer(open(csv_path, "w"))
for key, val in dict.items():
    w.writerow([key, val])
##########################################################################################

##########################################################################################

##########################################################################################
