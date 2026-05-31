import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

from Evaluation import evaluation


def image_to_spd_batch(images):
    spd_mats = []
    for img in images:
        X = img.reshape(-1, img.shape[-1]).astype(np.float64)
        X = X - np.mean(X, axis=0)
        cov = np.dot(X.T, X) / (X.shape[0] - 1)
        cov += 1e-5 * np.eye(cov.shape[0])  # ensure SPD
        spd_mats.append(cov)
    return np.array(spd_mats)


#  BiMap Layer
def BiMap(X, W):
    WT = tf.transpose(W)
    return tf.matmul(tf.matmul(WT, X), W)


#  ReEig Layer
def ReEig(X):
    eigvals, eigvecs = tf.linalg.eigh(X)
    eigvals = tf.maximum(eigvals, 1e-4)
    return tf.matmul(eigvecs,
                     tf.matmul(tf.linalg.diag(eigvals),
                               tf.transpose(eigvecs)))


#  LogEig Layer
def LogEig(X):
    eigvals, eigvecs = tf.linalg.eigh(X)
    logvals = tf.math.log(eigvals)
    return tf.matmul(eigvecs,
                     tf.matmul(tf.linalg.diag(logvals),
                               tf.transpose(eigvecs)))


#  SPDNet Model
def SPDNet_Model(spd_dim, num_classes):

    inputs = layers.Input(shape=(spd_dim, spd_dim))

    # Trainable BiMap weights
    W1 = tf.Variable(tf.random.normal([spd_dim, 16], dtype=tf.float64))
    W2 = tf.Variable(tf.random.normal([16, 8], dtype=tf.float64))

    x = BiMap(inputs, W1)
    x = ReEig(x)

    x = BiMap(x, W2)
    x = ReEig(x)

    x = LogEig(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def Model_SPDNet(train_images, train_labels, test_images, test_labels):
    Train_SPD = image_to_spd_batch(train_images)
    Test_SPD = image_to_spd_batch(test_images)
    num_classes = train_labels.shape[1]
    Train_Y = to_categorical(train_labels, num_classes)
    Test_Y = to_categorical(test_labels, num_classes)
    spd_dim = Train_SPD.shape[1]

    model = SPDNet_Model(spd_dim, num_classes)
    model.summary()

    model.fit(Train_SPD, Train_Y, epochs=50, batch_size=16)

    pred = model.predict(Test_SPD)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(pred, test_labels)
    return Eval, pred