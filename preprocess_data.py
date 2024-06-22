import cv2
import pickle
import os
import numpy as np
import tensorflow as tf
from collections import namedtuple
import random

project_dir = "segmentation/"
data_dir = "data/"

# (NOTE! this is taken from the official Cityscapes scripts:)
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

# (NOTE! this is taken from the official Cityscapes scripts:)
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      19 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      19 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      19 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# create a function mapping id to trainId:
id_to_trainId = {label.id: label.trainId for label in labels}
id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

new_img_height = 512 # (the height all images fed to the model will be resized to)
new_img_width = 1024 # (the width all images fed to the model will be resized to)
no_of_classes = 20 # (number of object classes (road, sidewalk, car etc.))

cityscapes_dir = data_dir + "cityscapes/"

train_imgs_dir = cityscapes_dir + "leftImg8bit/train/"
train_gt_dir = cityscapes_dir + "gtFine/train/"

val_imgs_dir = cityscapes_dir + "leftImg8bit/val/"
val_gt_dir = cityscapes_dir + "gtFine/val/"

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
            "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
            "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
            "bremen/", "bochum/", "aachen/"]
val_dirs = ["frankfurt/", "munster/", "lindau/"]


# Get the path to all training images and their corresponding label image
train_img_paths = []
train_trainId_label_paths = []

for dir_step, dir_name in enumerate(train_dirs):
    img_dir = os.path.join(train_imgs_dir, dir_name)
    file_names = os.listdir(img_dir)

    for step, file_name in enumerate(file_names):
        if step % 10 == 0:
            print(f"train dir {dir_step}/{len(train_dirs)-1}, step {step}/{len(file_names)-1}")

        img_id = file_name.split("_left")[0]

        # Read the image
        img_path = os.path.join(img_dir, file_name)
        img = cv2.imread(img_path, -1)

        # Resize the image without interpolation and save
        img_small = cv2.resize(img, (new_img_width, new_img_height), interpolation=cv2.INTER_NEAREST)
        img_small_path = os.path.join(train_imgs_dir, f"{img_id}.png")
        cv2.imwrite(img_small_path, img_small)
        train_img_paths.append(img_small_path)

        # Read and resize the corresponding label image without interpolation
        gt_img_path = os.path.join(train_gt_dir, dir_name, f"{img_id}_gtFine_labelIds.png")
        gt_img = cv2.imread(gt_img_path, -1)
        gt_img_small = cv2.resize(gt_img, (new_img_width, new_img_height), interpolation=cv2.INTER_NEAREST)

        # Convert the label image from id to trainId pixel values
        trainId_label = id_to_trainId_map_func(gt_img_small)

        # Save the label image
        trainId_label_path = os.path.join(project_dir, "data", f"{img_id}_trainId_label.png")
        cv2.imwrite(trainId_label_path, trainId_label)
        train_trainId_label_paths.append(trainId_label_path)

# Compute the mean color channels of the train images
print("Computing mean color channels of the train images")
no_of_train_imgs = len(train_img_paths)
mean_channels = np.zeros((3,))
for step, img_path in enumerate(train_img_paths):
    if step % 100 == 0:
        print(step)

    img = cv2.imread(img_path, -1)
    mean_channels += np.mean(img, axis=(0, 1))

mean_channels /= float(no_of_train_imgs)
pickle.dump(mean_channels, open(os.path.join(project_dir, "data", "mean_channels.pkl"), "wb"))

# Compute the class weights
print("Computing class weights")
trainId_to_count = {trainId: 0 for trainId in range(no_of_classes)}

# Get the total number of pixels in all train labels that are of each object class
for step, trainId_label_path in enumerate(train_trainId_label_paths):
    if step % 100 == 0:
        print(step)

    # Read the label image
    trainId_label = cv2.imread(trainId_label_path, -1)

    for trainId in range(no_of_classes):
        # Count how many pixels in the label image are of object class trainId
        trainId_mask = np.equal(trainId_label, trainId)
        label_trainId_count = np.sum(trainId_mask)

        # Add to the total count
        trainId_to_count[trainId] += label_trainId_count

# Compute the class weights according to the paper
class_weights = []
total_count = sum(trainId_to_count.values())
for trainId, count in trainId_to_count.items():
    trainId_prob = float(count) / float(total_count)
    trainId_weight = 1 / np.log(1.02 + trainId_prob)
    class_weights.append(trainId_weight)

# Save to disk
pickle.dump(class_weights, open(os.path.join(project_dir, "data", "class_weights.pkl"), "wb"))

# Get the path to all validation images and their corresponding label image
val_img_paths = []
val_trainId_label_paths = []

for dir_step, dir_name in enumerate(val_dirs):
    img_dir = os.path.join(val_imgs_dir, dir_name)
    file_names = os.listdir(img_dir)

    for step, file_name in enumerate(file_names):
        if step % 10 == 0:
            print(f"val dir {dir_step}/{len(val_dirs)-1}, step {step}/{len(file_names)-1}")

        img_id = file_name.split("_left")[0]

        # Read the image
        img_path = os.path.join(img_dir, file_name)
        img = cv2.imread(img_path, -1)

        # Resize the image without interpolation and save
        img_small = cv2.resize(img, (new_img_width, new_img_height), interpolation=cv2.INTER_NEAREST)
        img_small_path = os.path.join(project_dir, "data", f"{img_id}.png")
        cv2.imwrite(img_small_path, img_small)
        val_img_paths.append(img_small_path)

        # Read and resize the corresponding label image without interpolation
        gt_img_path = os.path.join(val_gt_dir, dir_name, f"{img_id}_gtFine_labelIds.png")
        gt_img = cv2.imread(gt_img_path, -1)
        gt_img_small = cv2.resize(gt_img, (new_img_width, new_img_height), interpolation=cv2.INTER_NEAREST)

        # Convert the label image from id to trainId pixel values
        trainId_label = id_to_trainId_map_func(gt_img_small)

        # Save the label image
        trainId_label_path = os.path.join(project_dir, "data", f"{img_id}_trainId_label.png")
        cv2.imwrite(trainId_label_path, trainId_label)
        val_trainId_label_paths.append(trainId_label_path)

# Save the validation data to disk
pickle.dump(val_trainId_label_paths, open(os.path.join(project_dir, "data", "val_trainId_label_paths.pkl"), "wb"))
pickle.dump(val_img_paths, open(os.path.join(project_dir, "data", "val_img_paths.pkl"), "wb"))

# Augment the train data by flipping all train images
no_of_train_imgs = len(train_img_paths)
print(f"number of train imgs before augmentation: {no_of_train_imgs}")

augmented_train_img_paths = []
augmented_train_trainId_label_paths = []

for step, (img_path, label_path) in enumerate(zip(train_img_paths, train_trainId_label_paths)):
    if step % 100 == 0:
        print(step)

    augmented_train_img_paths.append(img_path)
    augmented_train_trainId_label_paths.append(label_path)

    # Read the image
    img = cv2.imread(img_path, -1)

    # Flip the image and save
    img_flipped = cv2.flip(img, 1)
    img_flipped_path = img_path.split(".png")[0] + "_flipped.png"
    cv2.imwrite(img_flipped_path, img_flipped)
    augmented_train_img_paths.append(img_flipped_path)

    # Read the corresponding label image
    label_img = cv2.imread(label_path, -1)

    # Flip the label image and save
    label_img_flipped = cv2.flip(label_img, 1)
    label_img_flipped_path = label_path.split(".png")[0] + "_flipped.png"
    cv2.imwrite(label_img_flipped_path, label_img_flipped)
    augmented_train_trainId_label_paths.append(label_img_flipped_path)

# Randomly shuffle the augmented train data
augmented_train_data = list(zip(augmented_train_img_paths, augmented_train_trainId_label_paths))
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)

# Save the augmented train data to disk
train_img_paths, train_trainId_label_paths = zip(*augmented_train_data)
pickle.dump(train_img_paths, open(os.path.join(project_dir, "data", "train_img_paths.pkl"), "wb"))
pickle.dump(train_trainId_label_paths, open(os.path.join(project_dir, "data", "train_trainId_label_paths.pkl"), "wb"))

no_of_train_imgs = len(train_img_paths)
print(f"number of train imgs after augmentation: {no_of_train_imgs}")

