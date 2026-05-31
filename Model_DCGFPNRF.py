import cv2
import numpy as np
from retinaface import RetinaFace
from deepface import DeepFace
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout

from Evaluation import net_evaluation


# DEFORMABLE CONVOLUTION BLOCK
def deformable_conv_block(filters):
    block = Sequential([
        # CONVOLUTION 1
        Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),

        # CONVOLUTION 2
        Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),

        # MAX POOLING
        MaxPooling2D(pool_size=(2, 2))])

    return block


# BUILD DC-GFPN-RF MODEL
def build_dc_gfpn_rf_model():
    model = Sequential([
        # INPUT SHAPE
        Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # BLOCK 1
        deformable_conv_block(64),
        # BLOCK 2
        deformable_conv_block(128),

        # BLOCK 3
        deformable_conv_block(256),

        # BLOCK 4
        deformable_conv_block(512),

        # BLOCK 5
        deformable_conv_block(512),

        # BLOCK 6
        deformable_conv_block(1024),

        # BLOCK 7
        deformable_conv_block(1024),

        # GLOBAL FEATURE EXTRACTION
        GlobalAveragePooling2D(),

        # DENSE LAYER 1
        Dense(4096, activation='relu'),
        Dropout(0.5),
        # DENSE LAYER 2
        Dense(2048, activation='relu'),
        Dropout(0.4),

        # DENSE LAYER 3
        Dense(1024, activation='relu'),
        Dropout(0.4),

        # DENSE LAYER 4
        Dense(512, activation='relu'),
        Dropout(0.3),

        # DENSE LAYER 5
        Dense(256, activation='relu'),
        Dropout(0.3),

        # DENSE LAYER 6
        Dense(128, activation='relu'),
        Dropout(0.2),

        # FINAL OUTPUT
        # 7 EMOTIONS
        Dense(7, activation='softmax')

    ])

    return model


# IMAGE ENHANCEMENT
def enhance_image(image):
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    # DENOISE
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # SHARPEN
    kernel = np.array([
        [0, -1, 0],

        [-1, 5, -1],

        [0, -1, 0]

    ])
    image = cv2.filter2D(image, -1, kernel)
    return image


# EMOTION DETECTION
def detect_emotions(image):
    predictions = DeepFace.analyze(image, actions=['emotion'], detector_backend='retinaface', enforce_detection=False,
                                   silent=True)

    # SINGLE FACE -> LIST
    if not isinstance(predictions, list):
        predictions = [predictions]

    return predictions


def draw_results(image, predictions):
    targets = []
    for face in predictions:
        region = face['region']

        x = region['x']

        y = region['y']

        w = region['w']

        h = region['h']

        dominant_emotion = face[
            'dominant_emotion'
        ]

        confidence = face[
            'emotion'
        ][dominant_emotion]

        targets.append(dominant_emotion.capitalize())
        # LABEL
        label = f"{dominant_emotion.capitalize()} : {confidence:.1f}%"
        # FACE RECTANGLE
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # PUT TEXT
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image, targets


def Model_DCGFPNRF(Image):
    # BUILD MODEL
    model = build_dc_gfpn_rf_model()
    # OUTPUT LISTS
    detected_images = []

    all_targets = []
    # LOOP THROUGH IMAGES
    for image in Image:
        # ENHANCE IMAGE
        enhanced_image = enhance_image(image)
        # DETECT EMOTIONS
        predictions = detect_emotions(enhanced_image)
        # DRAW RESULTS
        detected_image, targets = draw_results(enhanced_image, predictions)
        # APPEND DETECTED IMAGE
        detected_images.append(detected_image)
        all_targets.append(targets)
    pred = model.predict(Image)
    Eval = net_evaluation(Image, detected_images)

    return Eval, pred, detected_images, all_targets
