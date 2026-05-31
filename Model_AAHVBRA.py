import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras import layers, models

from Evaluation import evaluation


#  Patch Embedding
def patch_embed(x, dim):
    x = layers.Conv2D(dim, 4, strides=4)(x)
    x = layers.Reshape((-1, dim))(x)
    return x


#  Spatial Bi-level Routing Attention
def BRA(sol, x, heads, dim):
    attn = layers.MultiHeadAttention(heads, dim)(x, x)
    x = layers.Dense(sol[0], activation='sigmoid')(x)
    route = layers.Dense(x.shape[-1], activation='sigmoid')(x)
    return x + attn * route


#  Transformer Block
def AAHV_block(sol, x, heads, dim, mlp):
    a = BRA(sol, x, heads, dim)
    x = layers.LayerNormalization()(x + a)

    m = layers.Dense(mlp, activation='gelu')(x)
    m = layers.Dense(x.shape[-1])(m)
    x = layers.LayerNormalization()(x + m)

    return x


#  Teacher
def TeacherNet(input_shape, classes):
    inp = layers.Input(input_shape)
    x = patch_embed(inp, 128)
    for _ in range(6):
        x = AAHV_block(x, 8, 64, 256)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(classes, activation='softmax')(x)
    return models.Model(inp, out)


#  Student
def StudentNet(sol, input_shape, classes):
    inp = layers.Input(input_shape)
    x = patch_embed(inp, 64)
    for _ in range(3):
        x = AAHV_block(sol, x, 4, 32, 128)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(classes, activation='softmax')(x)
    return models.Model(inp, out)


def Model_AAHVBRA(TrainData, TrainTarget, TestData, TestTarget, sol=None):
    if sol is None:
        sol = [5, 0.01, 5]
    IMG_SIZE = 32
    Train_X = np.zeros((TrainData.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(TrainData.shape[0]):
        temp = np.resize(TrainData[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((TestData.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(TestData.shape[0]):
        temp = np.resize(TestData[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    num_classes = TrainTarget.shape[1]
    input_shape = (Train_X.shape[1], Train_X.shape[2], Train_X.shape[3])

    teacher = TeacherNet(input_shape, num_classes)
    student = StudentNet(sol, input_shape, num_classes)

    teacher.compile(optimizer=Adam(learning_rate=sol[1]),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # Train teacher
    teacher.fit(Train_X, TrainTarget, epochs=sol[2], verbose=1)

    teacher.trainable = False
    optimizer = tf.keras.optimizers.Adam()

    # Knowledge Distillation Training
    for epoch in range(10):
        for i in range(len(Train_X)):
            x = Train_X[i:i + 1]
            y = TrainTarget[i:i + 1]

            with tf.GradientTape() as tape:
                t_pred = teacher(x, training=False)
                s_pred = student(x, training=True)

                hard = tf.keras.losses.categorical_crossentropy(y, s_pred)
                soft = tf.keras.losses.KLDivergence()(
                    tf.nn.softmax(t_pred / 3.0),
                    tf.nn.softmax(s_pred / 3.0)
                )
                loss = 0.7 * hard + 0.3 * soft

            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))

    pred = student.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(pred, TestTarget)
    return Eval, pred
