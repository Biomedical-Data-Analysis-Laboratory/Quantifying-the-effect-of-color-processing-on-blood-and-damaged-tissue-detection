""" Authored by: Neel Kanwal (neel.kanwal0@gmail.com)"""

# This file inference functions for models trained using main.py.
# Update paths to dataset and directories of best model weights.

import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import json
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models
import torchvision.transforms as transforms
from my_functions import  convert_batch_list, predict_cnn,  FocalLoss, epoch_test
from sklearn.metrics import accuracy_score, f1_score,  matthews_corrcoef, roc_auc_score
from scikitplot.metrics import plot_roc, plot_precision_recall, plot_lift_curve

# Select 
torch.cuda.empty_cache()
cuda_device = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #  limit GPU memory to avoid occupying all

NUM_WORKER = 16  # Number of simultaneous compute tasks == number of physical cores
BATCH_SIZE = 32
dropout = 0.2
torch.manual_seed(1700)

architectures = ["VGG16", "GoogleNet", "MobileNet", "ResNet", "DenseNet"]

artifact = "blood"  # "damaged" # Select processed dataset where to run inference
mod = "VGG16"  # "MobileNet" # Select model to intialize and load weights

# Best models from Table 1 in the paper; VGG16 did best for blood detection and MobileNet did best for damaged tissue detection.
if mod == "VGG16":
    weights = "/03_03_2022 11:17:20"
elif mod == "MobileNet":
    weights = "/03_04_2022 11:02:54"

else:
    print("Wrong ask")
    raise AssertionError

criterion = FocalLoss()

save_loc = "/path_to/IVMSP/experiments/"
if artifact == "damage":
    location_of_data = "/path_to/artifact_dataset/damage"
elif artifact == "blood":
    location_of_data = "/path_to/artifact_dataset/blood"

test_compose = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

t = time.time()

test_images = datasets.ImageFolder(root= location_of_data + "/test", transform=  test_compose)
idx2class = {v: k for k, v in test_images.class_to_idx.items()}
num_classes = len(test_images.classes)
test_loader = DataLoader(test_images, batch_size=BATCH_SIZE, shuffle=False, sampler=None, num_workers=NUM_WORKER, pin_memory=True)
classes_list = test_loader.dataset.classes
print(f"Length of test {len(test_images)} with {num_classes} classes")
print(f"Total data loading time in minutes: {(time.time() - t)/60:.3f}")


# for mod in architectures:
print(f"Working on {mod} models for {artifact}")
t = time.time()

if mod == "DenseNet":
    print("Loading DenseNet161 Model...............")
    model = models.densenet161(pretrained=True) # growth_rate = 48, num_init_features= 96, config = (6,12,36,24)
    num_features = model.classifier.in_features # 2208 --> less than 256
    model.classifier = custom_classifier(num_features, num_classes, dropout=dropout)
    best_model_wts = save_loc + mod + weights
    model.load_state_dict(torch.load(best_model_wts + "/best_weights.dat")['model'])

elif mod == "GoogleNet":
    print("Loading GoogleNet Model...............")
    model = models.googlenet(pretrained=True)
    num_features = model.fc.in_features
    model.fc = custom_classifier(num_features, num_classes, dropout=dropout)
    best_model_wts = save_loc + mod + weights
    model.load_state_dict(torch.load(best_model_wts + "/best_weights.dat")['model'])

elif mod == "ResNet":
    print("Loading ResNet152 Model...............")
    model = models.resnet152(pretrained=True)
    num_features = model.fc.in_features
    model.fc = custom_classifier(num_features, num_classes, dropout=dropout)
    best_model_wts = save_loc + mod + weights
    model.load_state_dict(torch.load(best_model_wts + "/best_weights.dat")['model'])

elif mod == "MobileNet":
    print("Loading MobileNet Model...............")
    model = models.mobilenet_v3_large(pretrained=True)
    model.classifier = custom_classifier(960, num_classes, dropout=dropout)
    best_model_wts = save_loc + mod + weights
    model.load_state_dict(torch.load(best_model_wts + "/best_weights.dat")['model'])
    # model.classifier[-1] = nn.Linear(1280, num_classes)

elif mod == "VGG16":
    print("Loading VGG16 Model...............")
    model = models.vgg16(pretrained=True)
    num_features = model.classifier[0].in_features
    model.classifier[6] = custom_classifier(4096, num_classes, dropout=dropout)
    best_model_wts = save_loc + mod + weights
    model.load_state_dict(torch.load(best_model_wts + "/best_weights.dat")['model'])

else:
    print("\nModel Does not exist")
    raise AssertionError

if torch.cuda.is_available():
    print("Cuda is available")# model should be on uda before selection of optimizer
    model = model.cuda()

print("\nTesting Starts....................")

path = os.path.join(best_model_wts, "prediction")
if not os.path.exists(path):
    os.mkdir(path)

# tr_acc, tr_loss = epoch_test(model, test_loader, criterion)
y_pred, y_true, probs, _ = predict_cnn(test_loader, model)
y_pred = convert_batch_list(y_pred)
y_true = convert_batch_list(y_true)
proba = convert_batch_list(probs)
# prob_for_one = [a[1] for a in proba]


accuracy = accuracy_score(y_true, y_pred)
print("Accuracy: ", accuracy)
f1 = f1_score(y_true, y_pred)
print("F1 Score: ", f1)
roc = roc_auc_score(y_true, y_pred)
print("ROC AUC Score: ", roc)
mathew_corr = matthews_corrcoef(y_true, y_pred)
print("Mathew Correlation Coefficient: ", mathew_corr)

plt.figure()
plot_roc(y_true,  proba, classes_to_plot= 1,  plot_micro=True, plot_macro=False, title=f'{mod} ROC Curve for {artifact}', title_fontsize='large')
plt.savefig(f"{path}/{mod} ROC Curve for {artifact}.png")

plt.clf()
plt.figure()
# lift: how much you gain by using your model over a random model for a given fraction of top scored predictions
plot_lift_curve(y_true,  proba, title=f'{mod} Lift Curve for {artifact}', title_fontsize='large')
plt.savefig(f"{path}/{mod} Lift Curve for {artifact}.png")

plt.clf()
plt.figure()
# KS plot helps to assess the separation between prediction distributions for positive and negative classes.
plot_ks_statistic(y_true, proba, title=f'{mod} KS Plot for {artifact}', title_fontsize='large')
plt.savefig(f"{path}/{mod} KS Plot for {artifact}.png")

plt.clf()
plt.figure()
# It is a curve that combines precision (PPV) and Recall (TPR) in a single visualization
# The higher on y-axis your curve is the better your model performance.
plot_precision_recall(y_true, proba, classes_to_plot = 1,plot_micro=True, title=f'{mod} Precision-Recall Curve for {artifact}',  title_fontsize='large')
plt.savefig(f"{path}/{mod}  Precision-Recall Curve for {artifact}.png")


print("--------------------------------------------")
print(f"Program finished for {mod}.......")
print("--------------------------------------------")
print(f"Total time in minutes: {(time.time() - t)/60:.3f}")