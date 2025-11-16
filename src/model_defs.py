import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_image_model(input_shape=(224,224,3), fine_tune_at=None):
    base = MobileNetV2(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    if fine_tune_at is not None:
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        base.trainable = False

    x = base.output
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base.input, outputs=out)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_video_model(frame_shape=(224,224,3), timesteps=25, cnn_trainable=False):
    cnn_base = MobileNetV2(include_top=False, input_shape=frame_shape, pooling='avg', weights='imagenet')
    cnn_base.trainable = cnn_trainable

    frame_input = layers.Input(shape=(timesteps,)+frame_shape)
    td = layers.TimeDistributed(cnn_base)(frame_input)
    td = layers.TimeDistributed(layers.Dense(256, activation='relu'))(td)

    x = layers.LSTM(128, return_sequences=False)(td)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=frame_input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model
