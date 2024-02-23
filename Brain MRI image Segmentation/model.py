
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
from keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

def unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        
    # Encoder (contracting path)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    drop1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(drop1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    drop2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(drop2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    drop3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(drop3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    drop4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(drop4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    drop5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(drop5)

    
    # Decoder (expansive path)
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use a lower learning rate for better convergence when dealing with small objects
    optimizer = Adam(learning_rate=1e-4)
    
    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model

# =============================================================================
# IMG_HEIGHT = 256
# IMG_WIDTH  = 256
# IMG_CHANNELS = 3
# 
# def get_model():
#     return unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
# 
# model = get_model()
# =============================================================================
