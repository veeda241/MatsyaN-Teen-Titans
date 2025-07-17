from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT
import numpy as np

def get_data_generators():
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT
    )
    train_gen = datagen.flow_from_directory(DATA_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
    val_gen = datagen.flow_from_directory(DATA_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')
    return train_gen, val_gen

def get_class_weights(train_generator):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    return dict(enumerate(class_weights))
