import numpy as np
import os
import pickle
import tensorflow as tf
import keras
from keras import layers
from keras import backend as K
from keras.layers import *
from keras.backend.tensorflow_backend import set_session
from keras.models import Model, load_model
from keras.layers import Concatenate
from keras import regularizers
import keras.layers.core as core
from keras.layers import Dense,Activation,Convolution2D, Convolution1D, MaxPool2D, Flatten, BatchNormalization, Dropout, Input, Bidirectional, MaxPool1D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
import math
import lightgbm as lgb
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class roc_callback(keras.callbacks.Callback):
    def __init__(self,training_data, validation_data):

        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):        
        y_pred = self.model.predict(self.x)
        roc = metrics.roc_auc_score(self.y, y_pred)      

        y_pred_val = self.model.predict(self.x_val)
        roc_val = metrics.roc_auc_score(self.y_val, y_pred_val)      

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return   
    
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    x = Convolution2D(filters, (num_row, num_col),
        kernel_initializer= 'glorot_normal',
        strides=strides,
        padding=padding,
        data_format='channels_last',
        use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation('relu')(x)
    #x = Dropout(0.3)(x)
    return x

def xceptiontest():
    #traindatapickle = open('/home/songjiazhi/atpcapsule/atp388/newfeature15.pickle','rb')
    #traindatapickle = open('/home/songjiazhi/atpcapsule/atp227/newfeature15.pickle','rb')
    #traindatapickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/label_train.pickle','rb')
    #traindatapickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/label_train.pickle','rb')
    traindatapickle = open('/home/songjiazhi/atpcapsule/paper/traindata/label.pickle','rb')
    traindata = pickle.load(traindatapickle)
    #label_train = traindata[0]
    label_train = traindata
    
    #pssmfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/pssmfeature.pickle','rb')
    #pssmfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/pssmfeature.pickle','rb')
    #pssmfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/pssmfeature_train.pickle','rb')
    #pssmfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/pssmfeature_train.pickle','rb')
    pssmfeature_train_pickle = open('/home/songjiazhi/atpcapsule/paper/traindata/pssmfeature.pickle','rb')
    pssmfeature_train = pickle.load(pssmfeature_train_pickle)
    #psipredfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/psipredfeature.pickle','rb')
    #psipredfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/psipredfeature.pickle','rb')
    #psipredfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/psipredfeature_train.pickle','rb')
    #psipredfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/psipredfeature_train.pickle','rb')
    psipredfeature_train_pickle = open('/home/songjiazhi/atpcapsule/paper/traindata/psipredfeature.pickle','rb')
    psipredfeature_train = pickle.load(psipredfeature_train_pickle)
    #chemicalfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/onehotfeature.pickle','rb')
    #chemicalfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/onehotfeature.pickle','rb')
    #chemicalfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/onehotfeature_train.pickle','rb')
    #chemicalfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/onehotfeature_train.pickle','rb')
    chemicalfeature_train_pickle = open('/home/songjiazhi/atpcapsule/paper/traindata/onehotfeature.pickle','rb')
    chemicalfeature_train = pickle.load(chemicalfeature_train_pickle)
    #multipssmfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/multipssmfeature.pickle','rb')
    #multipssmfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/multipssmfeature.pickle','rb')
    #multipssmfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/multipssmfeature_train.pickle','rb')
    #multipssmfeature_train_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/multipssmfeature_train.pickle','rb')
    #multipssmfeature_train = pickle.load(multipssmfeature_train_pickle)
    
    testdatapickle = open('/home/songjiazhi/atpcapsule/atp41/newfeature15.pickle','rb')
    #testdatapickle = open('/home/songjiazhi/atpcapsule/atp17/newfeature15.pickle','rb')
    #testdatapickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/label_test.pickle','rb')
    #testdatapickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/label_test.pickle','rb')
    testdata = pickle.load(testdatapickle)
    label_test = testdata[0]
    #label_test = testdata
    
    pssmfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp41/seperatefeature/pssmfeature.pickle','rb')
    #pssmfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp17/seperatefeature/pssmfeature.pickle','rb')
    #pssmfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/pssmfeature_test.pickle','rb')
    #pssmfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/pssmfeature_test.pickle','rb')
    pssmfeature_test = pickle.load(pssmfeature_test_pickle)
    psipredfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp41/seperatefeature/psipredfeature.pickle','rb')
    #psipredfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp17/seperatefeature/psipredfeature.pickle','rb')
    #psipredfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/psipredfeature_test.pickle','rb')
    #psipredfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/psipredfeature_test.pickle','rb')
    psipredfeature_test = pickle.load(psipredfeature_test_pickle)
    chemicalfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp41/seperatefeature/onehotfeature.pickle','rb')
    #chemicalfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp17/seperatefeature/onehotfeature.pickle','rb')
    #chemicalfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/onehotfeature_test.pickle','rb')
    #chemicalfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/onehotfeature_test.pickle','rb')
    chemicalfeature_test = pickle.load(chemicalfeature_test_pickle)
    #multipssmfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp41/seperatefeature/multipssmfeature.pickle','rb')
    #multipssmfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp17/seperatefeature/multipssmfeature.pickle','rb')
    #multipssmfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/multipssmfeature_test.pickle','rb')
    #multipssmfeature_test_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/multipssmfeature_test.pickle','rb')
    #multipssmfeature_test = pickle.load(multipssmfeature_test_pickle)

    pssmfeature_train = np.array(pssmfeature_train)
    pssmfeature_train = pssmfeature_train.reshape(-1,17,20,1)
    psipredfeature_train = np.array(psipredfeature_train)
    psipredfeature_train = psipredfeature_train.reshape(-1,17,3,1)
    chemicalfeature_train = np.array(chemicalfeature_train)
    chemicalfeature_train = chemicalfeature_train.reshape(-1,17,7,1)
    #multipssmfeature_train = np.array(multipssmfeature_train)
    #multipssmfeature_train = multipssmfeature_train.reshape(-1,17,40,1)
    label_train_one = np_utils.to_categorical(label_train, num_classes=2)
    
    pssmfeature_test = np.array(pssmfeature_test)
    pssmfeature_test = pssmfeature_test.reshape(-1,17,20,1)
    psipredfeature_test = np.array(psipredfeature_test)
    psipredfeature_test = psipredfeature_test.reshape(-1,17,3,1)
    chemicalfeature_test = np.array(chemicalfeature_test)
    chemicalfeature_test = chemicalfeature_test.reshape(-1,17,7,1)
    #multipssmfeature_test = np.array(multipssmfeature_test)
    #multipssmfeature_test = multipssmfeature_test.reshape(-1,17,40,1)
    label_test_one = np_utils.to_categorical(label_test, num_classes=2)   
    #feature_train = np.array(feature_train)
    #feature_train = feature_train.reshape(-1,17,71,1)
    #feature_test = np.array(feature_test)
    #feature_test = feature_test.reshape(-1,17,71,1)
    #label_train_one = np_utils.to_categorical(label_train, num_classes=2)
    #label_test_one = np_utils.to_categorical(label_test, num_classes=2)
    
    pssminput = Input((17,20,1))
    psipredinput = Input((17,3,1))
    chemicalinput = Input((17,7,1)) 
    #multipssminput = Input((17,40,1))
    
    #pssmfeature
    pssm_x = conv2d_bn(pssminput, 64, 3, 3)
    
    pssm_residual = Conv2D(128, (1,1), padding='same', data_format='channels_last')(pssm_x)
    pssm_residual = BatchNormalization(axis=3)(pssm_residual)
    
    pssm_x = SeparableConvolution2D(128, (3,3), padding='same', data_format='channels_last')(pssm_x)
    pssm_x = BatchNormalization(axis=3)(pssm_x)
    pssm_x = Activation('relu')(pssm_x)
    pssm_x = SeparableConvolution2D(128, (3,3), padding='same', data_format='channels_last')(pssm_x)
    pssm_x = BatchNormalization(axis=3)(pssm_x)
    
    pssm_x = layers.add([pssm_x, pssm_residual])
    
    pssm_residual = Conv2D(128, (1,1), padding='same', data_format='channels_last')(pssm_x)
    pssm_residual = BatchNormalization(axis=3)(pssm_residual)
    
    pssm_x = Activation('relu')(pssm_x)
    pssm_x = SeparableConvolution2D(128, (3,3), padding='same', data_format='channels_last')(pssm_x)
    pssm_x = BatchNormalization(axis=3)(pssm_x)
    pssm_x = Activation('relu')(pssm_x)
    pssm_x = SeparableConvolution2D(128, (3,3), padding='same', data_format='channels_last')(pssm_x)
    pssm_x = BatchNormalization(axis=3)(pssm_x)
    
    pssm_x = layers.add([pssm_x, pssm_residual])
    
    pssm_x = SeparableConvolution2D(128, (3,3), padding='same', data_format='channels_last')(pssm_x)
    pssm_x = BatchNormalization(axis=3)(pssm_x)
    pssm_x = Activation('relu')(pssm_x)
    pssm_x = Flatten()(pssm_x)
    pssm_x = Dense(256,activation='relu')(pssm_x)
    pssm_x = Dropout(0.4)(pssm_x)
    pssm_x = Dense(128, activation='relu')(pssm_x)
    
    #psipredfeature
    psipred_x = conv2d_bn(psipredinput, 64, 3, 3)
    psipred_x = SeparableConvolution2D(128, (3,3), padding='same', data_format='channels_last')(psipred_x)
    psipred_x = BatchNormalization(axis=3)(psipred_x)
    psipred_x = Activation('relu')(psipred_x)
    psipred_x = Flatten()(psipred_x)
    psipred_x = Dense(128, activation='relu')(psipred_x)
    
    #chemicalfeature
    chemical_x = conv2d_bn(chemicalinput, 64, 3, 3)
    
    chemical_residual = Conv2D(128, (1,1), padding='same', data_format='channels_last')(chemical_x)
    chemical_residual = BatchNormalization(axis=3)(chemical_residual)
    
    chemical_x = SeparableConvolution2D(128, (3,3), padding='same', data_format='channels_last')(chemical_x)
    chemical_x = BatchNormalization(axis=3)(chemical_x)
    chemical_x = Activation('relu')(chemical_x)
    chemical_x = SeparableConvolution2D(128, (3,3), padding='same', data_format='channels_last')(chemical_x)
    chemical_x = BatchNormalization(axis=3)(chemical_x)
    
    chemical_x = layers.add([chemical_x, chemical_residual])
    
    chemical_residual = Conv2D(128, (1,1), padding='same', data_format='channels_last')(chemical_x)
    chemical_residual = BatchNormalization(axis=3)(chemical_residual)
    
    chemical_x = Activation('relu')(chemical_x)
    chemical_x = SeparableConvolution2D(128, (3,3), padding='same', data_format='channels_last')(chemical_x)
    chemical_x = BatchNormalization(axis=3)(chemical_x)
    chemical_x = Activation('relu')(chemical_x)
    chemical_x = SeparableConvolution2D(128, (3,3), padding='same', data_format='channels_last')(chemical_x)
    chemical_x = BatchNormalization(axis=3)(chemical_x)
    
    chemical_x = layers.add([chemical_x, chemical_residual])
    
    chemical_x = SeparableConvolution2D(128, (3,3), padding='same', data_format='channels_last')(chemical_x)
    chemical_x = BatchNormalization(axis=3)(chemical_x)
    chemical_x = Activation('relu')(chemical_x)
    chemical_x = Flatten()(chemical_x)
    chemical_x = Dense(256, activation='relu')(chemical_x)
    chemical_x = Dropout(0.4)(chemical_x)
    chemical_x = Dense(128, activation='relu')(chemical_x)
    
    ##multipssmfeature
    #multipssm_x = conv2d_bn(multipssminput, 64, 3, 3)
    
    #multipssm_residual = Conv2D(128, (1,1),  padding='same', data_format='channels_last')(multipssm_x)
    #multipssm_residual = BatchNormalization(axis=3)(multipssm_residual)
    
    #multipssm_x = SeparableConv2D(128, (3,3), padding='same', data_format='channels_last')(multipssm_x)
    #multipssm_x = BatchNormalization(axis=3)(multipssm_x)
    #multipssm_x = Activation('relu')(multipssm_x)
    #multipssm_x = SeparableConv2D(128, (3,3), padding='same', data_format='channels_last')(multipssm_x)
    #multipssm_x = BatchNormalization(axis=3)(multipssm_x)
    
    #multipssm_x = layers.add([multipssm_x, multipssm_residual])
    ##multipssm_x = senet(multipssm_x, reduction=16)
    
    #multipssm_residual = Conv2D(128, (1,1), padding='same', data_format='channels_last')(multipssm_x)
    #multipssm_residual = BatchNormalization(axis=3)(multipssm_residual)
    
    #multipssm_x = Activation('relu')(multipssm_x)
    #multipssm_x = SeparableConv2D(128, (3,3), padding='same', data_format='channels_last')(multipssm_x)
    #multipssm_x = BatchNormalization(axis=3)(multipssm_x)
    #multipssm_x = Activation('relu')(multipssm_x)
    #multipssm_x = SeparableConv2D(128, (3,3), padding='same', data_format='channels_last')(multipssm_x)
    #multipssm_x = BatchNormalization(axis=3)(multipssm_x)
    
    #multipssm_x = layers.add([multipssm_x, multipssm_residual])
    ##multipssm_x = senet(multipssm_x, reduction=16)
    
    #multipssm_x = SeparableConv2D(128, (3,3), padding='same', data_format='channels_last')(multipssm_x)
    #multipssm_x = BatchNormalization(axis=3)(multipssm_x)
    #multipssm_x = Activation('relu')(multipssm_x)
    #multipssm_x = Flatten()(multipssm_x)
    #multipssm_x = Dense(256,activation='relu')(multipssm_x)
    #multipssm_x = Dropout(0.4)(multipssm_x)
    #multipssm_x = Dense(128,activation='relu')(multipssm_x)    
    
    #output = Concatenate(axis=1)([pssm_x, psipred_x, chemical_x, multipssm_x])
    output = Concatenate(axis=1)([pssm_x, psipred_x, chemical_x])
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(64, activation='relu')(output)
    output = Dense(2,activation='softmax')(output)   
    
    #model = Model(inputs=[pssminput, psipredinput, chemicalinput, multipssminput], outputs = output)
    model = Model(inputs=[pssminput, psipredinput, chemicalinput], outputs=output)
    adam = Adam(lr=0.0001,epsilon=1e-08)  
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])       
    #filepath = '/home/songjiazhi/atpcapsule/atp388/multiincepmodel2/weights-{epoch:02d}.hdf5'
    #filepath = '/home/songjiazhi/atpcapsule/atp227/multiresnextmodel/weights-{epoch:02d}.hdf5' 
    #filepath = '/home/songjiazhi/atpcapsule/atp388/fivefold/5/multixceptionmodel/weights-{epoch:02d}.hdf5'
    #filepath = '/home/songjiazhi/atpcapsule/atp227/fivefold/5/multixceptionmodel/weights-{epoch:02d}.hdf5'
    filepath = '/home/songjiazhi/atpcapsule/paper/multixception/weights-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath, save_best_only=False, save_weights_only=False)     
    #class_weight = {0:0.5205,1:12.7113}
    #class_weight = {0:0.5199,1:13.0584}
    class_weight = {0:0.5205, 1:12.7114}
    early_stopping = EarlyStopping(monitor='val_loss', patience=8)
    #model.fit([pssmfeature_train, psipredfeature_train, chemicalfeature_train, multipssmfeature_train], label_train_one, epochs=60, batch_size=256, shuffle=True, class_weight=class_weight, callbacks=[roc_callback(training_data=([pssmfeature_train, psipredfeature_train, chemicalfeature_train, multipssmfeature_train], label_train_one), validation_data=([pssmfeature_test, psipredfeature_test, chemicalfeature_test, multipssmfeature_test], label_test_one)), checkpoint])    
    model.fit([pssmfeature_train, psipredfeature_train, chemicalfeature_train], label_train_one, epochs=60, batch_size=256, shuffle=True, class_weight=class_weight, callbacks=[roc_callback(training_data=([pssmfeature_train, psipredfeature_train, chemicalfeature_train], label_train_one), validation_data=([pssmfeature_test, psipredfeature_test, chemicalfeature_test], label_test_one)), checkpoint, early_stopping])    
    
xceptiontest()
