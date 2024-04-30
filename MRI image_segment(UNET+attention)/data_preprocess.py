# =============================================================================
#                make image and mask folder and convet tif to png
# =============================================================================
import os
import glob
import pathlib
import shutil


DATA_ROOT = "./kaggle_3m/"
image_paths = []

# including all subdirectories
for path in glob.glob(DATA_ROOT + "**/*_mask.tif"):
    def strip_base(p):
        parts = pathlib.Path(p).parts
        return os.path.join(*parts[-2:])
    
    image = path.replace("_mask", "")
    if os.path.isfile(image):
        image_paths.append((strip_base(image), strip_base(path)))
    else:
        print("MISSING: ", image, "==>", path)

# Define directories for images and masks
images_dir = os.path.join(DATA_ROOT, "images")
masks_dir = os.path.join(DATA_ROOT, "masks")

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)


import cv2
# Move image and mask files to their respective directories and convert to PNG
for image_path, mask_path in image_paths:
    # Full path for source files
    src_image = os.path.join(DATA_ROOT, image_path)
    src_mask = os.path.join(DATA_ROOT, mask_path)
    
    # Read the image and mask using OpenCV
    image = cv2.imread(src_image, cv2.IMREAD_UNCHANGED) 
    mask = cv2.imread(src_mask, cv2.IMREAD_UNCHANGED)
    
    # Define the destination path for images and masks with .png extension
    dst_image = os.path.join(images_dir, os.path.basename(image_path).replace(".tif", ".png"))
    dst_mask = os.path.join(masks_dir, os.path.basename(mask_path).replace(".tif", ".png"))
    
    # Save the files in PNG format
    cv2.imwrite(dst_image, image)
    cv2.imwrite(dst_mask, mask)


# Loop through the items in the directory
for item in os.listdir(DATA_ROOT):
    item_path = os.path.join(DATA_ROOT, item)  # Get the full path of the item
    
    # Check if the current item is a directory and not 'images' or 'masks'
    if os.path.isdir(item_path) and item not in ['images', 'masks']:
        shutil.rmtree(item_path)  # Remove the folder and all its contents
