from model import unet_model
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


image_directory = './data/images/'
mask_directory = './data/masks/'


SIZE = 256
image_dataset = []  
mask_dataset = []  

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'png'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, cv2.IMREAD_UNCHANGED)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(mask_directory+image_name , cv2.IMREAD_UNCHANGED)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

#Normalize images
image_dataset = np.array(image_dataset) / 255.
#Do not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims(np.array(mask_dataset) / 255, -1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (SIZE, SIZE,3)), cmap='BrBG_r')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (SIZE, SIZE, 1)), cmap='gray')
plt.show()

###############################################################
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

def get_model():
    return unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()

history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    batch_size=4,
                    verbose=1,
                    epochs=50, 
                    shuffle=False, 
                    #callbacks=callbacks
                    )


############################################################
#Save the model wiight

model.save('weight.hdf5')

############################################################
#Evaluate the model


#plot the training and validation accuracy and loss at each epoch
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


##################################
#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#######################################################################
#Predict on a few images
model.load_weights('weight.hdf5') 


# Assuming X_test contains RGB images with shape (num_samples, height, width, 3)
test_img_number = random.randint(0, len(X_test) - 1)  # Adjusted for zero indexing
print("Showing output for image patch Number:", test_img_number)

test_img = X_test[test_img_number]  # This is already an RGB image
ground_truth = y_test[test_img_number]

# Since the image is already in RGB, no need to modify its channels
test_img_input = np.expand_dims(test_img, 0)  # Add batch dimension

# Get prediction
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

# Plotting
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)  # Removed cmap='gray' to show RGB image
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')  # Assuming the ground truth is still grayscale
plt.subplot(233)
plt.title('Prediction on "test image"')
plt.imshow(prediction, cmap='gray')  # Prediction is typically binary mask, shown in grayscale
