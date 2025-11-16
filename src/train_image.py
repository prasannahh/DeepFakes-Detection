import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from model_defs import build_image_model

DATA_DIR = r"D:\Prasanna\archive"   # must contain 'real' and 'fake' subfolders
BATCH = 16
IMG_SIZE = (224,224)

def main():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # match inference normalization
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='binary', subset='training', shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='binary', subset='validation', shuffle=False
    )

    print("Class indices:", train_gen.class_indices)

    model = build_image_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), fine_tune_at=None)
    model.summary()

    model.fit(train_gen, validation_data=val_gen, epochs=12)
    model.save("image_model_final.h5")
    print("✅ Saved image_model.h5")

if __name__ == "__main__":
    main()
