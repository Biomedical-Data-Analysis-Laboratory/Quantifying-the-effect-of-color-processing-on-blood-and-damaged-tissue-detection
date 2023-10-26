"""
Authored by: Neel Kanwal (neel.kanwal0@gmail.com)

This script can be applied on patched dataset to obtain color-processed versions.

Example: Stain normalize with Vahadane algorithm a list of H&E images.
Paper: "Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images", Vahadane et al, 2016.

Example: Stain normalize with Macenko algorithm a list of H&E images.
Paper: "A METHOD FOR NORMALIZING HISTOLOGY SLIDES FOR QUANTITATIVE ANALYSIS", Macenko et al, 2009.

"""

import os
from PIL.Image import open, fromarray
import numpy as np
import time
from tqdm import tqdm
from histocartography.preprocessing import MacenkoStainNormalizer
from histocartography.preprocessing import VahadaneStainNormalizer
# ref_image_path = None
dataset = "blood" # blood , #damage # Choose the dataset if your artifact_dataset has different directories.
normalizer = "vahadane" # macenko, vahadane # Choose color normalization scheme
os.chdir('/home/neel/')
test = True
valid = True

if normalizer == "vahadane":
    # normalization = VahadaneStainNormalizer(lambda_s=0.2, threshold=0.5)
    normalization = VahadaneStainNormalizer()
    print("Stain Matrix for Vahadane\n")
    print(normalization.stain_matrix_target)
elif normalizer == "macenko":
    normalization = MacenkoStainNormalizer()
    print("Stain Matrix for Macenko\n")
    print(normalization.stain_matrix_target)

new_path ="artifact_dataset/" + dataset + "_" + normalizer

if test:
    os.mkdir(os.path.join(new_path, "test"))
    os.mkdir(os.path.join(new_path, "test", dataset))
    os.mkdir(os.path.join(new_path, "test", "artifact_free"))
else:
    print("Directory exists")

tr_artifact_free = os.listdir(os.path.join("artifact_dataset/" + dataset, "training/", "artifact_free"))
tr_artifact = os.listdir(os.path.join("artifact_dataset/" + dataset, "training/", dataset))
val_artifact_free = os.listdir(os.path.join("artifact_dataset/" + dataset, "validation/", "artifact_free"))
val_artifact = os.listdir(os.path.join("artifact_dataset/" + dataset, "validation/", dataset))

if test:
    test_artifact_free = os.listdir(os.path.join("artifact_dataset/" + dataset, "test/", "artifact_free"))
    test_artifact = os.listdir(os.path.join("artifact_dataset/" + dataset, "test/", dataset))

    test_images_af = [a for a in test_artifact_free if a.endswith("png")]
    test_images_a = [a for a in test_artifact if a.endswith("png")]

print(f"###### Normalizing {dataset} dataset with {normalizer} Scheme ##########")
t = time.time()
print("\nTotal files in training ", len(train_images_af) + len(train_images_a))

for img in tqdm(train_images_a):
    img_copy= open(os.path.join("artifact_dataset", dataset, "training", dataset, img)).convert('RGB')
    # img_copy= open(os.path.join("artifact_dataset", dataset, "training", dataset, img))
    target = np.array(img_copy)
    norm_img = normalization._normalize_image(target)
    # norm_img = normalization.process(target)
    sav_img = fromarray(np.uint8(norm_img)).convert("RGBA")
    # sav_img = fromarray(np.uint8(norm_img))
    sav_img.save(os.path.join(new_path, "training", dataset, img))
print(f"Finished with {dataset} folder of training dataset.\n")

for img in tqdm(train_images_af):
    img_copy= open(os.path.join("artifact_dataset", dataset, "training", "artifact_free", img)).convert('RGB')
    # img_copy= open(os.path.join("artifact_dataset", dataset, "training", "artifact_free", img))
    target = np.array(img_copy)
    norm_img = normalization._normalize_image(target)
    # norm_img = normalization.process(target)
    sav_img = fromarray(np.uint8(norm_img)).convert("RGBA")
    # sav_img = fromarray(np.uint8(norm_img))
    sav_img.save(os.path.join(new_path, "training", "artifact_free", img))

print("Finished with artifact free folder training dataset.\n")
print("\nTotal files in validation ", len(val_images_af) + len(val_images_a))

for img in tqdm(val_images_a):
    img_copy= open(os.path.join("artifact_dataset", dataset, "validation", dataset, img)).convert('RGB')
    target = np.array(img_copy)
    norm_img = normalization._normalize_image(target)
    # norm_img = normalization.process(target)
    sav_img = fromarray(np.uint8(norm_img)).convert("RGBA")
    sav_img.save(os.path.join(new_path, "validation", dataset, img))
print(f"Finished with {dataset} folder of validation dataset.\n")

for img in tqdm(val_images_af):
    img_copy= open(os.path.join("artifact_dataset", dataset, "validation", "artifact_free", img)).convert('RGB')
    target = np.array(img_copy)
    norm_img = normalization._normalize_image(target)
    # norm_img = normalization.process(target)
    sav_img = fromarray(np.uint8(norm_img)).convert("RGBA")
    sav_img.save(os.path.join(new_path, "validation", "artifact_free", img))

print("Finished with artifact free folder of validation dataset.\n")

if test:
    print("\nTotal files in test ", len(test_images_af) + len(test_images_a))

    for img in tqdm(test_images_a):
        img_copy= open(os.path.join("artifact_dataset", dataset, "test", dataset, img)).convert('RGB')
        target = np.array(img_copy)
        norm_img = normalization._normalize_image(target)
        sav_img = fromarray(np.uint8(norm_img)).convert("RGBA")
        sav_img.save(os.path.join(new_path, "test", dataset, img))
    print(f"Finished with {dataset} folder of test dataset.\n")

    for img in tqdm(test_images_af):
        img_copy= open(os.path.join("artifact_dataset", dataset, "test", "artifact_free", img)).convert('RGB')
        target = np.array(img_copy)
        norm_img = normalization._normalize_image(target)
        sav_img = fromarray(np.uint8(norm_img)).convert("RGBA")
        sav_img.save(os.path.join(new_path, "test", "artifact_free", img))

print(f"Total colour normalization time for {normalizer} in minutes: {(time.time() - t)/60:.3f}")
