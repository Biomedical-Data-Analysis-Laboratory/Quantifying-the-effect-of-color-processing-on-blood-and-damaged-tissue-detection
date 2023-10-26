""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file will run inference and create ROC plots usinf best models on blood and damaged tissue, trained using main.py.
# Update paths to dataset and directories of best model weights.


import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models
import torchvision.transforms as transforms
from my_functions import convert_batch_list
from my_functions import  predict_cnn,  custom_classifier, FocalLoss, make_roc
import numpy as np
import matplotlib.pyplot as plt



torch.cuda.empty_cache()
cuda_device = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)


NUM_WORKER = 16  # Number of simultaneous compute tasks == number of physical cores
BATCH_SIZE = 32
dropout = 0.2
torch.manual_seed(1700)

vgg16_weights_blood = "/03_03_2022 11:17:20"
vgg16_weights_damage = "/03_04_2022 10:39:39"
mobilenet_weights_damage = "/03_04_2022 11:02:54"
mobilenet_weights_blood =  "/03_03_2022 12:32:10"

criterion = FocalLoss()

save_loc = "/path_to/IVMSP/experiments/"
location_dam_data = "/path_to/artifact_dataset/damage"
location_blood_data = "/path_to/artifact_dataset/blood"

test_compose = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

t = time.time()

test_images_dam = datasets.ImageFolder(root= location_dam_data + "/test", transform= test_compose)
idx2class_dam = {v: k for k, v in test_images_dam.class_to_idx.items()}
num_classes_dam = len(test_images_dam.classes)
test_loader_dam = DataLoader(test_images_dam, batch_size=BATCH_SIZE, shuffle=False, sampler=None, num_workers=NUM_WORKER, pin_memory=True)
classes_list_dam = test_loader_dam.dataset.classes
print(f"Length of damaged tissue test {len(test_images_dam)} with {num_classes_dam} classes")

test_images_blood = datasets.ImageFolder(root= location_blood_data + "/test", transform= test_compose)
idx2class_blood = {v: k for k, v in test_images_blood.class_to_idx.items()}
num_classes_blood = len(test_images_blood.classes)
test_loader_blood = DataLoader(test_images_blood, batch_size=BATCH_SIZE, shuffle=False, sampler=None, num_workers=NUM_WORKER, pin_memory=True)
classes_list_blood = test_loader_blood.dataset.classes
print(f"Length of blood {len(test_images_blood)} with {num_classes_blood} classes")
print(f"Total data loading time in minutes: {(time.time() - t)/60:.3f}")


print("Loading MobileNet Model for damage...............")
model_mobilenet_dam = models.mobilenet_v3_large(pretrained=True)
model_mobilenet_dam.classifier = custom_classifier(960, num_classes_dam, dropout=dropout)
best_model_wts = save_loc + "MobileNet" + mobilenet_weights_damage
model_mobilenet_dam.load_state_dict(torch.load(best_model_wts + "/best_weights.dat")['model'])

print("Loading MobileNet Model for blood...............")
model_mobilenet_bl = models.mobilenet_v3_large(pretrained=True)
model_mobilenet_bl.classifier = custom_classifier(960, num_classes_dam, dropout=dropout)
best_model_wts = save_loc + "MobileNet" + mobilenet_weights_blood
model_mobilenet_bl.load_state_dict(torch.load(best_model_wts + "/best_weights.dat")['model'])


print("Loading VGG16 Model for blood..................")
model_vgg16_bl= models.vgg16(pretrained=True)
num_features = model_vgg16_bl.classifier[0].in_features
model_vgg16_bl.classifier[6] = custom_classifier(4096, num_classes_blood, dropout=dropout)
best_model_wts = save_loc + "VGG16" + vgg16_weights_blood
model_vgg16_bl.load_state_dict(torch.load(best_model_wts + "/best_weights.dat")['model'])

print("Loading VGG16 Model for damage..................")
model_vgg16_dam = models.vgg16(pretrained=True)
num_features = model_vgg16_dam.classifier[0].in_features
model_vgg16_dam.classifier[6] = custom_classifier(4096, num_classes_blood, dropout=dropout)
best_model_wts = save_loc + "VGG16" + vgg16_weights_damage
model_vgg16_dam.load_state_dict(torch.load(best_model_wts + "/best_weights.dat")['model'])



if torch.cuda.is_available():
    print("Cuda is available")# model should be on uda before selection of optimizer
    model_vgg16_bl = model_vgg16_bl.cuda()
    model_vgg16_dam = model_vgg16_dam.cuda()
    model_mobilenet_dam = model_mobilenet_dam.cuda()
    model_mobilenet_bl = model_mobilenet_bl.cuda()
print("\nTesting Starts....................")


# for mobileNet damaged
y_pred_mob_dam, y_true_mob_dam, probs_mob_dam, _ = predict_cnn(test_loader_dam, model_mobilenet_dam)
y_pred_mob_dam = convert_batch_list(y_pred_mob_dam)
y_true_mob_dam = convert_batch_list(y_true_mob_dam)
probs_mob_dam = convert_batch_list(probs_mob_dam)
#for mobilenet blood
y_pred_mob_bl, y_true_mob_bl, probs_mob_bl, _ = predict_cnn(test_loader_blood, model_mobilenet_bl)
y_pred_mob_bl = convert_batch_list(y_pred_mob_bl)
y_true_mob_bl = convert_batch_list(y_true_mob_bl)
probs_mob_bl = convert_batch_list(probs_mob_bl)

# for vgg16 blood
y_pred_vgg_bl, y_true_vgg_bl, probs_vgg_bl, _ = predict_cnn(test_loader_blood, model_vgg16_bl)
y_pred_vgg_bl = convert_batch_list(y_pred_vgg_bl)
y_true_vgg_bl = convert_batch_list(y_true_vgg_bl)
probs_vgg_bl = convert_batch_list(probs_vgg_bl)

# for vgg16 damaged
y_pred_vgg_dam, y_true_vgg_dam, probs_vgg_dam, _ = predict_cnn(test_loader_dam, model_vgg16_dam)
y_pred_vgg_dam = convert_batch_list(y_pred_vgg_dam)
y_true_vgg_dam = convert_batch_list(y_true_vgg_dam)
probs_vgg_dam = convert_batch_list(probs_vgg_dam)


plt.figure(1).clf()

ax = make_roc(y_true_mob_dam,  probs_mob_dam, classes_to_plot=1, cmap="Accent", name='MobileNet', artifact='damaged tissue', title='ROC Curves for MobileNet and VGG16', title_fontsize='large', text_fontsize="small")
ax = make_roc(y_true_mob_bl,  probs_mob_bl, classes_to_plot=1, cmap="flag", name='MobileNet', artifact='blood', title='ROC Curves for MobileNet and VGG16', title_fontsize='large', text_fontsize="small", ax=ax)

last = make_roc(y_true_vgg_dam,  probs_vgg_dam, classes_to_plot=1, cmap="inferno", name='VGG16', artifact='damaged tissue', title='ROC Curves for MobileNet and VGG16', title_fontsize='large', text_fontsize="small", ax=ax)
ax = make_roc(y_true_vgg_bl,  probs_vgg_bl, classes_to_plot=1, cmap="viridis", name='VGG16', artifact='blood', title='ROC Curves for MobileNet and VGG16', title_fontsize='large', text_fontsize="small", ax=last)

plt.savefig(f"ROC_PLOT.png")

print(".......Finished........")