# Dog Breed Image Classifier

A **production** end-to-end deep-learning computer-vision system that classifies
photos of dogs into **120 breeds** using **transfer learning** with TensorFlow/Keras.

> *Given a photo of a dog, which of 120 breeds is it?*

This is a **multi-class image classification** problem. The goal is to beat the
original [Stanford Dogs research paper](http://vision.stanford.edu/aditya86/ImageNetDogs/)
baseline (~22% mean per-class accuracy) using a pretrained convolutional neural
network and transfer learning.

## ⚠️ Runtime: designed for Google Colab (GPU)

Unlike a typical local project, these notebooks are built to run **end-to-end on
[Google Colab](https://colab.research.google.com/)** for free GPU access, and they
**download their datasets at runtime** (the image datasets are hundreds of MB and
are *not* bundled in this repo). Training on a CPU is impractical — a GPU is
strongly recommended.

`pyproject.toml` is included for reference / local experimentation, but running
locally requires your own TensorFlow-GPU setup.

## Project structure

```
dog-breed-image-classifier/
├── README.md
├── pyproject.toml
├── data/                          # saved prediction results (outputs)
│   ├── dog-vision-full-model-predictions-with-mobilenetV2.csv
│   └── dog-vision-prediction-probabilites-array.csv
├── images/
│   ├── dog-photo-1..4.jpeg        # custom demo photos
│   ├── dog-photos.zip
│   └── unstructured-data-*.png    # diagrams referenced by the main notebook
└── notebooks/
    ├── end-to-end-dog-vision-v2.ipynb     # main (detailed) notebook
    └── end-to-end-dog-vision-video.ipynb  # condensed walkthrough
```

Two notebook variants are included:

- **`...-v2.ipynb`** — the full, detailed notebook. Uses the **Stanford Dogs
  dataset** (20,000+ images, 120 breeds), builds an image pipeline with
  `tf.keras.utils.image_dataset_from_directory`, fine-tunes an **EfficientNetV2**
  model via transfer learning, evaluates per-class accuracy, predicts on custom
  images, and ends with a Gradio demo.
- **`...-video.ipynb`** — a condensed, streamlined walkthrough. Uses the **Kaggle
  Dog Breed Identification** dataset with a **MobileNetV2 + TensorFlow Hub**
  feature extractor. (This variant is written for the Colab + Google Drive
  workflow.)

## The data

- **v2 — Stanford Dogs:** downloaded at runtime from
  [vision.stanford.edu](http://vision.stanford.edu/aditya86/ImageNetDogs/)
  (`images.tar`, `annotation.tar`, `lists.tar`), then split into train/test folders.
- **video — Kaggle Dog Breed Identification:** the
  [Kaggle competition dataset](https://www.kaggle.com/c/dog-breed-identification/data)
  (requires a Kaggle account), read from a mounted Google Drive folder.

The two CSVs under `data/` are **saved prediction outputs** from earlier runs,
kept as reference results. Downloaded datasets, extracted image folders, and
trained models (`*.keras`, `*.h5`) are git-ignored to keep the repo light.

## Approach

The project follows a 6-step machine learning framework:

1. **Problem definition** — classify dog photos into 120 breeds (multi-class).
2. **Data** — Stanford Dogs / Kaggle Dog Breed images, loaded as batched,
   preprocessed `tf.data` datasets.
3. **Evaluation** — beat the original paper's ~22% mean per-class accuracy.
4. **Features** — learned automatically by the network (deep learning).
5. **Modelling** — transfer learning from a pretrained CNN (EfficientNetV2 /
   MobileNetV2), then fine-tuning.
6. **Experiments** — compare training on 10% vs. the full dataset, evaluate
   per-class accuracy, inspect the "most wrong" predictions, and predict on
   custom images.

## Getting started

1. Open one of the notebooks in **Google Colab** (set the runtime to a **GPU**):
   - `notebooks/end-to-end-dog-vision-v2.ipynb` (detailed)
   - `notebooks/end-to-end-dog-vision-video.ipynb` (condensed walkthrough)
2. Run the cells top to bottom. Each notebook downloads its dataset on first run.

To experiment locally instead, install the dependencies with
[uv](https://docs.astral.sh/uv/) (`uv sync`) and use a GPU-enabled TensorFlow
build — note that training the full model without a GPU is impractical.
