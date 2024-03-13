# Quantifying-the-effect-of-color-processing-on-blood-and-damaged-tissue-detection
This repository contains Source code for the paper "Quantifying the effect of color processing on blood and damaged tissue detection in whole slide images", published at 14th IEEE  Image, Video, and Multidimensional Signal Processing Workshop (IVMSP 2022).

Published Version = https://ieeexplore.ieee.org/abstract/document/9816283
# Requirements 
- Histocartography
- Histolab
- Pytorch
- Scikit-learn
- Pandas
- Numpy
  
# Abstract 
Histological tissue examination has been a longstanding practice for cancer diagnosis, where pathologists identify the presence of tumors on glass slides. Slides acquired from laboratory routines may contain unintentional artifacts due to complications in surgical resection. Blood and damaged tissue artifacts are two common problems associated with transurethral resection of the bladder tumor. Differences in histotechnical procedures among laboratories may also result in color variations and minor inconsistencies in outcome. A digitized version of a glass slide known as a whole slide image (WSI) holds enormous potential for automated diagnostics. The presence of irrelevant areas in a WSI undermines diagnostic value for pathologists as well as computational pathology (CPATH) systems. Therefore, automatic detection and exclusion of diagnostically irrelevant areas may lead to more reliable predictions. In this paper, we are detecting blood and damaged tissue against diagnostically relevant tissue. We gauge the effectiveness of transfer learning against training from scratch. The best models give 0.99 and 0.89 F1 scores for blood and damaged tissue detection. Since blood and damaged tissue have subtle color differences, we assess the impact of color processing methods on the binary classification performance of five well-known architectures. Finally, we remove the color to understand its importance against morphology on classification performance.

<img width="620" alt="image" src="https://github.com/NeelKanwal/Quantifying-the-effect-of-color-processing-on-blood-and-damaged-tissue-detection/assets/52494244/e2f273c3-a2b8-4b8b-81e0-d90d2d76e2a9">

# Results
<img width="610" alt="image" src="https://github.com/NeelKanwal/Quantifying-the-effect-of-color-processing-on-blood-and-damaged-tissue-detection/assets/52494244/fb4ccbab-38f3-4626-8209-e764627ec1e3">

<img width="610" alt="image" src="https://github.com/NeelKanwal/Quantifying-the-effect-of-color-processing-on-blood-and-damaged-tissue-detection/assets/52494244/ee56d7e7-1907-4f9e-a21c-ab7abfa334a2">

# Dataset
The dataset is publicaly available at Zenodo. https://zenodo.org/records/10809442. 

You can use D40x directory and corresponding folders with artifacts to organize in the following order.

If you plan to use your dataset, then organize it with the following structure: An example of a blood artifact is shown below.

<img width="228" alt="image" src="https://github.com/NeelKanwal/Quantifying-the-effect-of-color-processing-on-blood-and-damaged-tissue-detection/assets/52494244/99a8a908-5da0-41a6-b280-ebe7864d6533">

```
- path_to\blood_dataset
      - training
           -- artifact_free
            -- blood
      - validation
            -- artifact_free
            -- blood
       - test
            -- artifact_free
            -- blood
```


# How to use the code
- Process directory contains code to create color processed version of dataset.
- Use main.py to select and train DCNN model. Hyperparametric selection can be defined inside the header of the file.
- Use Inference.py to update path to experiment directory and test on new data.
  
If you use this code or build on top of it, then please cite our paper.
```
@inproceedings{kanwal2022quantifying,
  title={Quantifying the effect of color processing on blood and damaged tissue detection in whole slide images},
  author={Kanwal, Neel and Fuster, Saul and Khoraminia, Farbod and Zuiverloon, Tahlita CM and Rong, Chunming and Engan, Kjersti},
  booktitle={2022 IEEE 14th Image, Video, and Multidimensional Signal Processing Workshop (IVMSP)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}
```
