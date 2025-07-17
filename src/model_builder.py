from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from config import IMAGE_SIZE

def build_model(num_classes):
    # Load MobileNetV2 base
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    # Freeze base layers (optional for transfer learning)
    for layer in base.layers:
        layer.trainable = False

    # Add custom head
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Final model
    model = Model(inputs=base.input, outputs=output)
    return model
