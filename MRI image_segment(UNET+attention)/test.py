import cv2
import numpy as np
from matplotlib import pyplot as plt
#from Normal_unet import unet_model
from model_senet import unet_with_senet

# change the value od threshold untill you get the highest IoU value  
thes = 0.3

def get_model():
    # Ensure your model is updated to handle 256x256x3 input for RGB images
    #return unet_model(256, 256, 3)  
    return unet_with_senet(256, 256, 3)

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    return round(iou, 3)

def dice_coef(y_true, y_pred):
    intersect = np.sum(y_pred * y_true)
    total_sum = np.sum(y_pred) + np.sum(y_true)
    dice = 2 * intersect / total_sum
    return round(dice, 3)  # Round up to 3 decimal places

def prediction(model, image):
    image_norm = np.array(image) / 255
    image_input = np.expand_dims(image_norm, 0)  # Add batch dimension
    prediction = (model.predict(image_input)[0,:,:,0] > thes).astype(np.uint8)     # change the value od threshold untill you get the highest IoU value 
    return prediction

# Load model
model = get_model()
#model.load_weights('weight_model_Unet.hdf5')
model.load_weights('weight_model_senet.hdf5')
# model.load_weights('weight_model_srm.hdf5')

# Load RGB image and corresponding grayscale mask
# =============================================================================
# test_image = cv2.imread('test_normal/image_10.png')  # Ensure this is an RGB image
# test_mask =  cv2.imread('test_normal/mask_10.png', cv2.IMREAD_GRAYSCALE)
# =============================================================================
test_image = cv2.imread('test_senet/image_10.png')  # Ensure this is an RGB image
test_mask =  cv2.imread('test_senet/mask_10.png', cv2.IMREAD_GRAYSCALE)

# Predict
segmented_image = prediction(model, test_image)

# Calculate IoU and Dice
iou = calculate_iou(test_mask > 0, segmented_image > 0)
dice = dice_coef(test_mask > 0, segmented_image > 0)

# Save and display images
# =============================================================================
# plt.imsave('./test_normal/pred_img_10.png', segmented_image, cmap='gray')
plt.imsave('./test_senet/pred_img_10.png', segmented_image, cmap='gray')
# plt.figure(figsize=(20, 10))
# plt.subplot(131)
# plt.title('Input Image')
# plt.imshow(large_image)
# plt.subplot(132)
# plt.title('Ground Truth Label')
# plt.imshow(large_mask, cmap='gray')
# plt.subplot(133)
# plt.title('Prediction on "Input image"')
# plt.imshow(segmented_image, cmap='gray')
# plt.show()
# =============================================================================

print("Intersection over Union (IoU):", iou)
print("Dice similarity coefficient (DSC):", dice)
