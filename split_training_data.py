import os
import shutil

coco_training_images = 82783
split = int(0.7 * 82783)

coco_images_folder = "D:\project\neural-vqa-tensorflow\Data\train2014"
supervised_folder = "supervised_folder"
unsupervised_folder = "unsupervised_folder" 

count = 0
for file in os.listdir(coco_file_path):
    if file.endswith(".jpg"):
        if count%3 < split:
            number = file.split('_')
            data1.append(int(number[2][:-4]))
            shutil.move(os.path.join(coco_images_folder, file), os.path.join(coco_images_folder, supervised_folder, file))
            count += 1
        else:
            number = file.split('_')
            data3.append(int(number[2][:-4]))
            shutil.move(os.path.join(coco_images_folder, file), os.path.join(coco_images_folder, unsupervised_folder, file))
            #count += 1
