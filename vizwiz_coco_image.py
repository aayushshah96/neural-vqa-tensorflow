import os
import shutil

'''
vizwiz format to COCO format convertor -- images only
Reads all unhidden {target_ext} images from {images_dir}
and moves them onto {target_dir} with a new COCO compatible
name. For eg. COCO_train2014_000000587432.jpg
Before running this file, make sure:
1) ./Data/vizwiz/Images exists with all vizwiz images in it
2) ./Data/train_2014 exists, can have original COCO images
3) Check target prefix, id, suffix etc values
Warning: Only tested for Linux till now.
'''
target_id = 581922                                  # Unique id used to start conversion process from
target_prefix = 'COCO_train2014_'                   # Prefix added to all images
target_suffix = ''                                  # Suffix added to all images
target_ext = 'jpg'                                  # File extension/type (used to filter input images)
target_dir = os.path.join(os.getcwd(),'Data','train_2014')       # Directory to move images onto

images_dir = os.path.join(os.getcwd(),'Data','vizwiz','Images')    # Directory to move images from
_, _, image_names = next(os.walk(images_dir))       # Get all image names


for name in image_names:
    # Ignore hidden files and non matching file extensions (target_ext)
    if name[0] != '.' and name.split('.')[1] == target_ext:
        new_name = target_prefix + str(target_id).zfill(12) + target_suffix + '.' + target_ext
        shutil.move(os.path.join(images_dir, name), os.path.join(target_dir, new_name))
        target_id += 1
