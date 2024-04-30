
from tensorflow.keras.layers import Dropout, MaxPooling2D, UpSampling2D, concatenate, GlobalAveragePooling2D, Dense, Reshape, multiply, Input, Conv2D
from tensorflow.keras.models import Model

# Define the SENet block 
def squeeze_excite_block(input_tensor, ratio=16):
    '''Create a squeeze and excitation block'''
    init = input_tensor
    channel_axis = -1
    filters = init.shape[channel_axis]

    se = GlobalAveragePooling2D()(init)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)

    x = multiply([init, se])
    return x

# Function to add a conv block with SE block
def conv_block(input_tensor, num_filters, se_block=True):
    x = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(input_tensor)
    x = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    if se_block:
        x = squeeze_excite_block(x)
    return x


def unet_with_senet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Encoder
    c1 = conv_block(inputs, 16, se_block=True)
    drop1 = Dropout(0.1)(c1)
    p1 = MaxPooling2D((2, 2))(drop1)
    
    c2 = conv_block(p1, 32, se_block=True)
    drop2 = Dropout(0.1)(c2)
    p2 = MaxPooling2D((2, 2))(drop2)
    
    c3 = conv_block(p2, 64, se_block=True)
    drop3 = Dropout(0.2)(c3)
    p3 = MaxPooling2D((2, 2))(drop3)
    
    c4 = conv_block(p3, 128, se_block=True)
    drop4 = Dropout(0.2)(c4)
    p4 = MaxPooling2D((2, 2))(drop4)
    
    # Bottleneck
    c5 = conv_block(p4, 256, se_block=True)
    c5 = Dropout(0.2)(c5)
    c5 = conv_block(c5, 256, se_block=True)

    # Decoder
    u1 = UpSampling2D((2, 2))(c5)
    merge1 = concatenate([u1, c4])
    drop5 = Dropout(0.2)(merge1)
    c6 = conv_block(drop5, 128, se_block=True)

    u2 = UpSampling2D((2, 2))(c6)
    merge2 = concatenate([u2, c3])
    drop6 = Dropout(0.2)(merge2)
    c7 = conv_block(drop6, 64, se_block=True)

    u3 = UpSampling2D((2, 2))(c7)
    merge3 = concatenate([u3, c2])
    drop7 = Dropout(0.1)(merge3)
    c8 = conv_block(drop7, 32, se_block=True)
    
    u4 = UpSampling2D((2, 2))(c8)
    merge4 = concatenate([u4, c1])
    drop8 = Dropout(0.1)(merge4)
    c9 = conv_block(drop8, 16, se_block=True)
    
    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(c9)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.summary()
    
    return model
