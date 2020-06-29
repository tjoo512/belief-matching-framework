import os
import argparse
import pandas as pd
import scipy.io as sio
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, UpSampling2D, Flatten, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from keras.datasets import cifar10
import tensorflow as tf
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser(description='Transfer learning from ResNet-50 pretrained on ImageNet')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset name (one of : cifar10, scar, food101)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU numbers for training')
parser.add_argument('--coeff', default=1e-2, type=float,
                    help='Coefficient to KL term in BM loss. Set -1 to use CrossEntropy Loss')
parser.add_argument('--prior', default=1.0, type=float,
                    help='Dirichlet prior parameter')
parser.add_argument('--save_path', default='result.pkl', type=str,
                    help='save target path')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def belief_matching_loss(targets, logits):
    beta_coeff = args.prior
    coeff = args.coeff
    alphas = tf.exp(logits)
    betas = tf.ones_like(logits) * beta_coeff
    a_zero = tf.reduce_sum(alphas, -1)

    a_self = tf.reduce_sum(tf.digamma(alphas)*targets, -1)
    ll_loss = a_self - tf.digamma(a_zero)
    loss1 = tf.lgamma(a_zero) - tf.reduce_sum(tf.lgamma(alphas), -1)
    loss2 = tf.reduce_sum(
                (alphas - betas) * (tf.digamma(alphas) - tf.digamma(tf.expand_dims(a_zero, -1))), -1)
    kl_loss = loss1 + loss2
    loss = coeff*kl_loss - ll_loss
    return tf.reduce_mean(loss)

def main():

    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in resnet_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    model = Sequential()

    if args.dataset == 'cifar10':
        model.add(UpSampling2D())
        model.add(UpSampling2D())
        model.add(UpSampling2D())

        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = preprocess_input(x_train)
        x_test = preprocess_input(x_test)
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)
    elif args.dataset == 'scar':
        num_classes = 196

        train_annos_path = './devkit/cars_train_annos.mat'
        test_annos_path = './devkit/cars_test_annos_withlabels.mat'
        classes_path = './devkit/cars_meta.mat'

        def get_labels(annos_path, classes_path):
            car_annos = sio.loadmat(path + annos_path)
            car_meta = sio.loadmat(path + classes_path)
            annotations = car_annos["annotations"][0,:]
            nclasses = len(car_meta["class_names"][0])
            class_names = dict(zip(range(1,nclasses),[c[0] for c in car_meta["class_names"][0]]))
            
            labelled_images = {}
            dataset = []
            for i,arr in enumerate(annotations):
                # the last entry in the row is the image name
                # The rest is the data, first bbox, then classid
                dataset.append([y[0][0] for y in arr][0:5]+[arr[5][0]])
            # Convert to a DataFrame, and specify the column names
            temp_df = pd.DataFrame(dataset, 
                            columns =['BBOX_X1','BBOX_Y1','BBOX_X2','BBOX_Y2','ClassID','filename'])

            temp_df = temp_df.assign(ClassName=temp_df.ClassID.map(dict(class_names)))
            temp_df.columns = ['bbox_x1','bbox_y1','bbox_x2','bbox_y2','class_id','filename', 'class_name']
            return temp_df

        path = './'
        train_df = get_labels(train_annos_path, classes_path)
        train_df['is_test'] = 0
        train_df.to_csv(path + 'train_labels.csv', index=False)

        test_df = get_labels(test_annos_path, classes_path)
        test_df['is_test'] = 1

        # Add missing class name! - 'smart fortwo Convertible 2012'
        train_df.loc[train_df['class_name'].isnull(), 'class_name'] = 'smart fortwo Convertible 2012'
        test_df.loc[test_df['class_name'].isnull(), 'class_name'] = 'smart fortwo Convertible 2012'

        frames = [train_df, test_df]
        labels_df = pd.concat(frames)
        labels_df.reset_index(inplace=True, drop=True)
        labels_df = labels_df[['filename', 'bbox_x1', 'bbox_y1','bbox_x2','bbox_y2',
                                    'class_id', 'class_name','is_test']]

        # adjust the test file names
        labels_df['filename'].loc[labels_df['is_test']==1] = 'test_' + labels_df['filename']

        # Add the cropped file names
        labels_df['filename_cropped'] = labels_df['filename'].copy()
        labels_df['filename_cropped'].loc[labels_df['is_test']==0] = 'cropped_' + labels_df['filename']

        labels_df.to_csv(path + 'labels_with_annos.csv')
        labels_df.head()

        train_df = labels_df.loc[labels_df['is_test'] == 0]
        train_df = train_df[['filename', 'class_name']]

        test_df = labels_df.loc[labels_df['is_test'] == 1]
        test_df = test_df[['filename', 'class_name']]

        test_df["filename"] = test_df["filename"].apply(lambda x: x.split('_')[-1])
        datagen = ImageDataGenerator()

        train_it = datagen.flow_from_dataframe(
                        dataframe=train_df,
                        directory="./cars_train/",
                        x_col="filename",
                        y_col="class_name",
                        batch_size=args.batch_size,
                        seed=42,
                        shuffle=True,
                        class_mode="categorical",
                        target_size=(224,224))

        test_it = datagen.flow_from_dataframe(
                        dataframe=test_df,
                        directory="./cars_test/",
                        x_col="filename",
                        y_col="class_name",
                        batch_size=args.batch_size,
                        seed=42,
                        shuffle=True,
                        class_mode="categorical",
                        target_size=(224,224))

    elif args.dataset == 'food101':
        num_classes = 101
        datagen = ImageDataGenerator()
        train_it = datagen.flow_from_directory('./food-101/images/', target_size=(224,224), batch_size=args.batch_size)
        test_it = datagen.flow_from_directory('./food-101/test/', target_size=(224,224), batch_size=args.batch_size)
    else:
        print("Invalid dataset.")
        exit()

    model.add(resnet_model)
    model.add(GlobalAveragePooling2D())

    if args.coeff < 0:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=args.lr),metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation=None))
        model.compile(loss=belief_matching_loss,optimizer=keras.optimizers.Adam(lr=args.lr),metrics=['accuracy'])

    if args.dataset == 'cifar10':
        history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_test, y_test))
    else:
        history = model.fit_generator(train_it,
                                    steps_per_epoch=len(train_it),
                                    epochs=args.epochs,
                                    validation_data=test_it)

    print("Maximum Test Accuracy : ", np.max(history.history['val_acc']))

    pickle.dump(history.history, open(args.save_path, 'wb'))

if __name__ == '__main__':
    main()
