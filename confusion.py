
import numpy as np
import tensorflow as tf
import cv2
import pickle as cPickle
from utilities import label_img_to_color
from model import ENet_model

# Load trained model
model_id = "1"
img_height = 512
img_width = 1024
batch_size = 4
no_of_classes = 20  # Assuming you have 20 classes
model = ENet_model(model_id, img_height=img_height, img_width=img_width, batch_size=batch_size)
saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), write_version=tf.compat.v1.train.SaverDef.V2)
sess = tf.compat.v1.Session()
saver.restore(sess, "training_logs/best_model/model_1_epoch_23.ckpt")  # Provide the path to your trained model

# Load validation data
val_img_paths = cPickle.load(open("segmentation/data/val_img_paths.pkl", "rb"))
val_trainId_label_paths = cPickle.load(open("segmentation/data/val_trainId_label_paths.pkl", "rb"))

train_mean_channels = cPickle.load(open("segmentation/data/mean_channels.pkl", "rb"))

# Define label conversion params
layer_idx = np.arange(img_height).reshape(img_height, 1)
component_idx = np.tile(np.arange(img_width), (img_height, 1))

# Initialize lists to store predicted and true labels
predicted_labels_list = []
true_labels_list = []

# Iterate over validation data
for img_path, label_path in zip(val_img_paths, val_trainId_label_paths):
    # Read image
    img = cv2.imread(img_path, -1)
    img = cv2.resize(img, (img_width, img_height))  # Resize image to match model input size
    img = img.astype(np.float32) - train_mean_channels  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Read true label
    trainId_label = cv2.imread(label_path, -1)
    onehot_label = np.zeros((img_height, img_width, no_of_classes), dtype=np.float32)
    onehot_label[layer_idx, component_idx, trainId_label] = 1
    
    # Run inference
    logits_output = sess.run(model.logits, feed_dict={model.imgs_ph: img,
                                                      model.early_drop_prob_ph: 0.5,
                                                      model.late_drop_prob_ph: 0.5})
    predicted_labels = np.argmax(logits_output, axis=3)
    
    # Append predicted and true labels
    predicted_labels_list.append(predicted_labels[0])
    true_labels_list.append(np.argmax(onehot_label, axis=2))

# Convert lists to numpy arrays
predicted_labels_array = np.array(predicted_labels_list)
true_labels_array = np.array(true_labels_list)

# Calculate pixel accuracy
pixel_acc = np.mean(predicted_labels_array == true_labels_array)

# Calculate mean Intersection over Union (mIoU)
intersection = np.sum(np.logical_and(predicted_labels_array, true_labels_array), axis=(1, 2))
union = np.sum(np.logical_or(predicted_labels_array, true_labels_array), axis=(1, 2))
iou = np.mean(intersection / union)

# Calculate class-wise Intersection over Union (IoU)
class_iou = intersection / union

# Print or save the results
print("Pixel Accuracy: ", pixel_acc)
print("mIoU: ", iou)
print("Class-wise IoU: ", class_iou)
