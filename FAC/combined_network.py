from __future__ import absolute_import
from __future__ import print_function
from math import sqrt

import keras.backend as K
import numpy as np
import tensorflow as tf
from import_data_facd import *
# from keras.layers import Input, Lambda, Dense
from keras.models import Model
# from tensorflow.keras.models import Model
from keras.optimizers import SGD
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras.regularizers import l2
# from googlenet import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from mini_model import create_mini_googlenet, GoogLeNet
from keras.applications.resnet50 import ResNet50
from googlenet_functional import *

path = "./FACD_image/"

def BTPred(scalars):
    """
    This is the output when we predict comparison labels. s1, s2 = scalars (beta.*x)
    """
    s1 = scalars[0]
    s2 = scalars[1]
    return s1 - s2

def scaledThurstoneLoss(alpha):
    def ThurstoneLoss(y_true, y_pred):
        '''P(si-sj|yij) = 0.5 * erfc(-yij*(si-sj) / sqrt(2))'''
        return - (1-alpha) * K.log(0.5 * tf.erfc(-y_true*y_pred / sqrt(2)))
    return ThurstoneLoss

def scaledBTLoss(alpha):
    def BTLoss(y_true, y_pred):
        """
        Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
        y_true:-1 or 1
        y_pred:si-sj
        alpha: 0-1
        """
        exponent = K.exp(-y_true * (y_pred))
        return (1-alpha) * K.log(1 + exponent)
    return BTLoss

def scaledCrossEntropy(alpha):
    """Use this crossentropy loss function when there are two or more label classes. 
    We expect labels to be provided in a one_hot representation. If you want to provide 
    labels as integers, please use SparseCategoricalCrossentropy loss. 
    There should be # classes floating point values per feature.
    """

    def crossEntropy(y_true, y_pred):
        return alpha * K.categorical_crossentropy(y_true, y_pred)
    return crossEntropy

def scaledHinge(alpha):
    def hinge(y_true, y_pred):
        return alpha * K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)
    return hinge

def scaledCompCrossEntropy(alpha):
    def crossEntropy(y_true, y_pred):
        return 1- alpha
    return tf.keras.losses.CategoricalCrossentropy()

def scaledCompHinge(alpha):
    def hinge(y_true, y_pred):
        return (1-alpha) * K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)
    return hinge

class combined_deep_ROP(object):
    """
    Training and testing the neural network with 5000 images.
    """
    def __init__(self, input_shape=(3, 224, 224), no_of_classes=2, data_dir=path):

        #HERE 1  
        self.data_gen = ImportData(input_shape=input_shape, data_dir=data_dir)
        self.input_shape = input_shape
        self.no_of_classes = no_of_classes
        self.data_dir = data_dir

    def create_siamese(self, reg_param=0.02, no_of_score_layers=1, max_no_of_nodes=128):
        print("FUNCTION: create_siamese")

        # get features from base network

        # MINI NET CODE:
        input1 = keras.Input(shape=self.input_shape)
        input2 = keras.Input(shape=self.input_shape)

        feature1=create_mini_googlenet(input1, "feature1")
        feature2 = create_mini_googlenet(input2, "feature2")
        score1 = feature1(input1)
        score2 = feature2(input2)

        for l in range(5):
            layer_l = layers.Dense(int(max_no_of_nodes / (l+4)), activation='relu', kernel_regularizer=l2(reg_param), name='score'+str(l))
            score1 = layer_l(score1)
            score2 = layer_l(score2)

        # create final layers of absolute and comparison
        abs_out = layers.Dense(self.no_of_classes, activation='softmax', kernel_regularizer=l2(reg_param), name='abs')
        comp_out = layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_param), name='comp')
        

        # absolute part
        abs_out1 = abs_out(score1)
        abs_net = Model(inputs=input1, outputs=abs_out1, name="AbsoluteNetwork")
        
        # comparison part
        comp_out1 = comp_out(score1)
        comp_out2 = comp_out(score2)

        distance = layers.Lambda(BTPred, output_shape=(1,), name="distance")([comp_out1, comp_out2])
        comp_net = keras.Model(inputs=[input1, input2], outputs=distance, name="ComparisonNetwork")

        # keras.utils.plot_model(comp_net, "comp_net.png", show_shapes=True)
        # keras.utils.plot_model(abs_net, "abs_net.png", show_shapes=True)

        comp_net = Model([input1, input2], distance)
        return abs_net, comp_net

    def train(self, save_model_name='./combined.h5',
              reg_param=0.002, no_of_score_layers=1, max_no_of_nodes=128, learning_rate=1e-2,
              abs_loss=scaledCrossEntropy, comp_loss=scaledCompCrossEntropy, alpha=0.5, epochs=100, batch_size=32):
        """
        Training CNN except validation and test folds
        """
        abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels = self.data_gen.load_data(datasplit='train')
        # print("comp_imgs_1", comp_imgs_1.shape)
        
        abs_net, comp_net = self.create_siamese(reg_param=reg_param, no_of_score_layers=no_of_score_layers, max_no_of_nodes=max_no_of_nodes)

        # print(abs_loss(0.5))
        abs_net.compile(loss=scaledCrossEntropy(alpha=alpha), optimizer=SGD(learning_rate), metrics=['acc'])
        comp_net.compile(loss=scaledCompCrossEntropy(alpha=alpha), optimizer=SGD(learning_rate), metrics=['acc'])

        # for layer in comp_net.layers: 
        #     print(layer.get_config(), layer.get_weights())
        # train on abs only, comp only or both
        for epoch in range(epochs):
            print("EPOCH: ", epoch)
            abs_net.fit(abs_imgs, abs_labels, batch_size=batch_size, epochs=1)
            comp_net.fit([comp_imgs_1, comp_imgs_2], comp_labels, batch_size=batch_size, epochs=1)
            print("COMPARISON NETWORK")
            comp_pred = comp_net.predict([comp_imgs_1, comp_imgs_2], verbose=True)
            comp_evaluate = comp_net.evaluate([comp_imgs_1, comp_imgs_2], comp_labels ,verbose=True)

            print("ABSOLUTE NETWORK")
            abs_pred = abs_net.predict(abs_imgs, verbose=True)
            abs_evaluate = abs_net.evaluate(abs_imgs, abs_labels, verbose=True)

        # Save weights
        abs_net.save('Abs_' + save_model_name)
        comp_net.save('Comp_' + save_model_name)

    def test(self, set, model_file,
              reg_param=0.0002, no_of_score_layers=1, max_no_of_nodes=128, learning_rate=1e-4,
              abs_loss=scaledCrossEntropy, comp_loss=scaledCompCrossEntropy, alpha=0.5):
        """
        Testing CNN on validation/test fold.
        Predict 0th class, the class with the highest score
        """
        abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels = self.data_gen.load_data(datasplit="test")
        abs_net, comp_net = self.create_siamese\
                    (reg_param=reg_param, no_of_score_layers=no_of_score_layers, max_no_of_nodes=max_no_of_nodes)
        # load weights and compile: BASE NETWORK HAS THE SAME WEIGHTS
        # make sure we have weights everytime we test. we have two single input-single output branches
        comp_test_model = Model(inputs=comp_net.input[0], outputs=comp_net.get_layer('comp').get_output_at(0))
        comp_test_model.load_weights('Comp_' + model_file, by_name=True)
        abs_net.load_weights('Abs_' + model_file, by_name=True)
        comp_net.load_weights('Comp_' + model_file, by_name=True)
        # compile all models
        comp_test_model.compile(loss=comp_loss(alpha=alpha), optimizer=SGD(learning_rate))
        abs_net.compile(loss=abs_loss(alpha=alpha), optimizer=SGD(learning_rate))
        comp_net.compile(loss=comp_loss(alpha=alpha), optimizer=SGD(learning_rate))
        #################TEST AUC, predict high quality!!!
        if alpha == 0.0:  # only comparison training, use comp models
            comp_pred = comp_net.predict([comp_imgs_1, comp_imgs_2])
            abs_pred = comp_test_model.predict(abs_imgs)
        elif alpha == 1.0:  # only absolute training, use abs models
            abs_pred = abs_net.predict(abs_imgs)[:, 1]
            comp_pred = abs_net.predict(comp_imgs_1)[:, 1] - abs_net.predict(comp_imgs_2)[:, 1]
        else:
            abs_pred = abs_net.predict(abs_imgs)[:, 1]
            comp_pred = comp_net.predict([comp_imgs_1, comp_imgs_2])
        with open(set+'_abs_auc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Abs loss: ' + str(abs_loss) + ' & Comp loss: ' + str(comp_loss))
            file.write('\n' + str(roc_auc_score(abs_labels, abs_pred)))
        with open(set + '_comp_auc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Abs loss: ' + str(abs_loss) + ' & Comp loss: ' + str(comp_loss))
            file.write('\n' + str(roc_auc_score(comp_labels, comp_pred)))
        with open(set+'_abs_prauc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Abs loss: ' + str(abs_loss) + ' & Comp loss: ' + str(comp_loss))
            file.write('\n' + str(average_precision_score(abs_labels, abs_pred)))
        with open(set + '_comp_prauc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Abs loss: ' + str(abs_loss) + ' & Comp loss: ' + str(comp_loss))
            file.write('\n' + str(average_precision_score(comp_labels, comp_pred)))
        #################TEST OTHER METRICS
        if alpha == 0.0:  # only comparison training, use comp models
            comp_pred = comp_net.predict([comp_imgs_1, comp_imgs_2])
            abs_pred = comp_test_model.predict(abs_imgs)
            # a scalar output, classify with threshold 0.5
            abs_pred_thresholded = (abs_pred > 0.5).astype(int)
            comp_pred_thresholded = 2 * (comp_pred > 0.0).astype(int) - 1
        elif alpha == 1.0:  # only absolute training, use abs models
            abs_pred = abs_net.predict(abs_imgs)
            comp_pred = abs_net.predict(comp_imgs_1)[:, 1] - abs_net.predict(comp_imgs_2)[:, 1]
            # a 2 class output, take the maximum, 1 if high quality
            abs_pred_012 = np.argmax(abs_pred, axis=1).astype(int)
            abs_pred_thresholded = (abs_pred_012 == 1).astype(int)
            comp_pred_thresholded = 2 * (comp_pred > 0.0).astype(int) - 1
        else:
            abs_pred = abs_net.predict(abs_imgs)
            comp_pred = comp_net.predict([comp_imgs_1, comp_imgs_2])
            # a 2 class output, take the maximum, 1 if high quality
            abs_pred_012 = np.argmax(abs_pred, axis=1).astype(int)
            abs_pred_thresholded = (abs_pred_012 == 1).astype(int)
            comp_pred_thresholded = 2 * (comp_pred > 0.0).astype(int) - 1
        # calculate absolute metrics
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for l in range(abs_labels.shape[0]):
            if abs_pred_thresholded[l] == 1 and abs_labels[l] == 1:
                TP += 1
            elif abs_pred_thresholded[l] == 1 and abs_labels[l] == 0:
                FP += 1
            elif abs_pred_thresholded[l] == 0 and abs_labels[l] == 0:
                TN += 1
            else:
                FN += 1
        N_pos_test = TP + FN
        N_neg_test = FP + TN
        if (TP + FP) > 0:
            precision = 1.0 * TP / (TP + FP)
        else:
            precision = 0
        if (TP + FN) > 0:
            recall = 1.0 * TP / (TP + FN)
        else:
            recall = 0
        abs_accuracy = 1.0 * (TP + TN) / (N_pos_test + N_neg_test)
        if precision == 0 and recall == 0:
            abs_f1 = 0
        else:
            abs_f1 = 2.0 * precision * recall / (precision + recall)
        # calculate comparison metrics
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for l in range(comp_labels.shape[0]):
            if comp_pred_thresholded[l] == 1 and comp_labels[l] == 1:
                TP += 1
            elif comp_pred_thresholded[l] == 1 and comp_labels[l] == -1:
                FP += 1
            elif comp_pred_thresholded[l] == -1 and comp_labels[l] == -1:
                TN += 1
            else:
                FN += 1
        N_pos_test = TP + FN
        N_neg_test = FP + TN
        if (TP + FP) > 0:
            precision = 1.0 * TP / (TP + FP)
        else:
            precision = 0
        if (TP + FN) > 0:
            recall = 1.0 * TP / (TP + FN)
        else:
            recall = 0
        comp_accuracy = 1.0 * (TP + TN) / (N_pos_test + N_neg_test)
        if precision == 0 and recall == 0:
            comp_f1 = 0
        else:
            comp_f1 = 2.0 * precision * recall / (precision + recall)
        # save results
        with open(set + '_abs_acc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Abs loss: ' + str(abs_loss) + ' & Comp loss: ' + str(comp_loss))
            file.write('\n' + str(abs_accuracy))
        with open(set + '_comp_acc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Abs loss: ' + str(abs_loss) + ' & Comp loss: ' + str(comp_loss))
            file.write('\n' + str(comp_accuracy))
        with open(set + '_abs_f1.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Abs loss: ' + str(abs_loss) + ' & Comp loss: ' + str(comp_loss))
            file.write('\n' + str(abs_f1))
        with open(set + '_comp_f1.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Abs loss: ' + str(abs_loss) + ' & Comp loss: ' + str(comp_loss))
            file.write('\n' + str(comp_f1))
            #################################################################







