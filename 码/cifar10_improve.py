import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dropout, Activation, GlobalAveragePooling2D,Conv2D
from time import process_time
from sklearn.model_selection import train_test_split

hang_begin = process_time()


def hang_net():
    hang_all_str = Sequential()

    hang_all_str.add(Conv2D(24, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 3)))
    hang_all_str.add(Activation('relu'))
    hang_all_str.add(Conv2D(24, kernel_size=3, strides=1, padding='same'))
    hang_all_str.add(Activation('relu'))
    hang_all_str.add(Conv2D(24, kernel_size=3, strides=2, padding='same'))
    hang_all_str.add(Dropout(0.5))

    hang_all_str.add(Conv2D(48, kernel_size=3, strides=1, padding='same'))
    hang_all_str.add(Activation('relu'))
    hang_all_str.add(Conv2D(48, kernel_size=3, strides=1, padding='same'))
    hang_all_str.add(Activation('relu'))
    hang_all_str.add(Conv2D(48, kernel_size=3, strides=2, padding='same'))
    hang_all_str.add(Dropout(0.5))

    hang_all_str.add(Conv2D(48, kernel_size=3, strides=1, padding='same'))
    hang_all_str.add(Activation('relu'))
    hang_all_str.add(Conv2D(48, kernel_size=1, strides=1, padding='valid'))
    hang_all_str.add(Activation('relu'))
    hang_all_str.add(Conv2D(10, kernel_size=1, strides=1, padding='valid'))

    hang_all_str.add(GlobalAveragePooling2D())
    hang_all_str.add(Activation('softmax'))

    hang_all_str.summary()
    return hang_all_str

hang_all_str=hang_net()

#set parameter class number,weight decay(wd),momentum(rush)
hang_kinds = 10
hang_iter = 1
hang_block = 64
hang_wd = 0
hang_rush = 0.93
hang_rate = 0.006

# Load the CIFAR10 data.
(hangtr, hangtr_label), (hangte, hangte_label) = cifar10.load_data()

#shrink train data
#hangtr_remain, hangtr, hangtr_label_remain, hangtr_label = \
    #train_test_split(hangtr, hangtr_label, test_size=12500, stratify=hangtr_label)
print("traning sample number:",hangtr.shape)
 
# data pre process.cahnge class vector to binary class matrix.
hangtr_label = keras.utils.to_categorical(hangtr_label, hang_kinds)
hangte_label = keras.utils.to_categorical(hangte_label, hang_kinds)
sgd = optimizers.SGD(lr=hang_rate, decay=hang_wd, momentum=hang_rush, nesterov=True)
hang_all_str.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# data Normalization.
hangtr = hangtr.astype('float32') / 255
hangte = hangte.astype('float32') / 255
hangtr_mean = np.mean(hangtr, axis=0)
hangtr -= hangtr_mean
hangte -= hangtr_mean

hang_stru = hang_all_str.fit(hangtr, hangtr_label,
                             validation_data=(hangte, hangte_label),
                             batch_size=hang_block,
                             epochs=hang_iter)

hang_stop = process_time()
print("train total time:", hang_stop - hang_begin)

hang_acc= hang_all_str.evaluate(hangte, hangte_label)

print('The accuracy of model is:', hang_acc[1])

trainAcc = hang_stru.history['accuracy']
testAcc = hang_stru.history['val_accuracy']
# train time stop

a = np.arange(0, hang_iter)
plt.figure()
plt.title('train and test accuracy')
plt.plot(a, trainAcc)
plt.plot(a, testAcc)
