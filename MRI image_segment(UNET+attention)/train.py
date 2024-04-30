from model_unet import unet_model
#from model_senet import unet_with_senet
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# Data directory
image_directory = './kaggle_3m/images/'
mask_directory = './kaggle_3m/masks/'

SIZE = 256
image_dataset = []  
mask_dataset = []  

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    
    if (image_name.split('.')[1] == 'png'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, cv2.IMREAD_UNCHANGED)
        image = Image.fromarray(image)
        #image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(mask_directory+image_name , cv2.IMREAD_UNCHANGED)
        image = Image.fromarray(image)
        #image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

#Normalize images and mask
image_dataset = np.array(image_dataset) / 255.
mask_dataset = np.expand_dims(np.array(mask_dataset) / 255, -1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.15, random_state = 0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

###############################################################
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

def get_model():
    return unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    #return unet_with_senet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = get_model()

#Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=8,
                    verbose=1,
                    epochs=50, 
                    shuffle=False
                    )

############################################################
#Save the model wiight

model.save('weight_model_unet.hdf5')
#model.save('weight_model_senet.hdf5')

############################################################
#Evaluate the model

#plot the training and validation loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



