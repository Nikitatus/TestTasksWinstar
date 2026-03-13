# Named entity recognition + image classification

Here is implemented a machine learning pipeline that takes an image of animal and a sentence about the image and classifies whether the user's sentence is true or false.

The pipeline consists of Natural Language Processing (NLP) part for extraction of animal names from the input sentence and Computer Vision (CV) part for classification of the animal in the image.

## Overview
1. **NLP Component (NER)**: A custom Named Entity Recognition model based on Hugging Face's Transformers (`google-bert/bert-base-cased`) fine-tuned to extract animal entities (`B-ANIMAL`). 
2. **CV Component**: An Image Classification model based on a pretrained ResNet-18 architecture, fine-tuned on the "Animals-10" dataset (classes: dog, cat, horse, sheep, elephant, squirrel, chicken, spider, cow, butterfly).
3. **Pipeline**: Combines both models. It takes an image and a sentence, extracts animal entities, classifies the image, and returns `True` if the entities match the image classification, and `False` otherwise.

## Datasets
- **Animals-10 Dataset**: A public dataset of animal pictures from Google Search containing 10 classes. Used for training the CV ResNet-18 model.
- **Custom NER Dataset**: Synthetic dataset created for the 10 animal classes and formatted in CoNLL format, used to fine-tune the BERT model.

## Setup
### Requirements
All required packages are listed in 

Install requirements via pip:
```bash
pip install -r requirements.txt
```

## Training the Models

### Training the Image Model
Ensure you have the image dataset and run:
```bash
python IMG_train.py --data_dir /path/to/animals-10-dataset
```
*(Model will be saved to `models/img_model/image_model.pt`)* or whatever path you specify in the script.

### Training the NER Model
Ensure you have the `.conll` formatted text dataset and run:
```bash
python NER_train.py --train_data_path /path/to/train.conll --val_data_path /path/to/val.conll
```
*(Model will be saved to `models/ner_model`)* or whatever path you specify in the script.

Of course, you can change the training parameters in the scripts, like epochs, batch size, learning rate, etc, like this:
```bash
python IMG_train.py --data_dir /path/to/animals-10-dataset --epochs 10 --batch_size 64 --learning_rate 1e-4
```
## Usage and Inference

### Pipeline Inference
To check if the text matches the image, run `pipeline.py`:
```bash
python pipeline.py --image "IMG_20210118_165825.jpg" --sentence "I see a cat and a dog here"
```

### Standalone Image Inference
```bash
python IMG_inference.py --image "IMG_20210118_165825.jpg"
```

### Standalone NER Inference
```bash
python NER_inference.py --sentence "I walked with my dog today"
```

## Results
- **Image Classification (ResNet-18)**: Achieved ~95% validation accuracy on the Animals-10 dataset after 3 epochs of fine-tuning.
- **Named Entity Recognition (BERT)**: Achieved ~100% accuracy and F1 score on the synthetic dataset after 1 epoch of training.
