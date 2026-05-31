import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Lambda, GRU
from tensorflow.keras.layers import Permute, GlobalMaxPool1D, Concatenate, Dense, BatchNormalization
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from Evaluation import evaluation


def TrainGRU(train_data, train_target):
    input_ = Input(shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    lambda_ = Lambda(lambda x: tf.squeeze(x, axis=3))(input_)

    lstm_ = Bidirectional(GRU(8, activation='relu', return_sequences=True))(lambda_)

    permute_ = Permute((2, 1), input_shape=(128, 128))(lambda_)
    lstm_2 = Bidirectional(GRU(8, activation='relu', return_sequences=True))(permute_)

    lstm_ = BatchNormalization()(lstm_)
    maxpool1 = GlobalMaxPool1D()(lstm_)

    lstm_2 = BatchNormalization()(lstm_2)
    maxpool2 = GlobalMaxPool1D()(lstm_2)

    concat_ = Concatenate(axis=1)([maxpool1, maxpool2])
    dense_1 = Dense(20, activation='relu')(concat_)
    output_ = Dense(train_target.shape[1], activation='softmax')(dense_1)

    model = Model(input_, output_)
    model.summary()
    early = EarlyStopping(patience=4)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_target, epochs=100, steps_per_epoch=100, callbacks=[early, reduce_lr])
    return model


def Model_GRU(train_data, train_target, test_data, test_target):
    IMG_SIZE = 32
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    model = TrainGRU(Train_X, train_target)
    pred = model.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(pred, test_target)
    return Eval, pred