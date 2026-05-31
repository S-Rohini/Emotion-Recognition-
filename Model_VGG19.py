import numpy as np
from keras.applications.vgg19 import VGG19
from keras import layers, models
import cv2 as cv
from Evaluation import evaluation


def VGG_19(Train_Data, Train_Target, Test_Data, Test_Target):
    # Load the VGG19 model pre-trained on ImageNet data
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the convolutional layers of the VGG19 model
    for layer in base_model.layers:
        layer.trainable = False

    # Create a custom model by adding additional layers on top of VGG19
    model = models.Sequential()

    # Add the VGG19 base model
    model.add(base_model)

    # Add additional convolutional and max pooling layers
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(Train_Target.shape[1], activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def Model_VGG19(train_data, train_target, test_data, test_target):
    IMG_SIZE = 32
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    model = VGG_19(Train_X, train_target, Test_X, test_target)

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    model.fit(x=Train_X, y=train_target, epochs=200, steps_per_epoch=100)
    pred = model.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(pred, test_target)
    return Eval, pred

