# Facial Recognition Application for Gender and Age Identification

## Project Overview

This project involves developing a facial recognition application that identifies gender and age from images.

## Datasets

### UTKFace Dataset

The primary dataset used in this project is the UTKFace dataset. UTKFace is a large-scale face dataset with a wide age range (from 0 to 116 years old). It consists of over 20,000 face images annotated with age, gender, and ethnicity.

- **Links to UTKFace**:
  - [GitHub](https://susanqq.github.io/UTKFace/)
  - [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
  - [Papers with Code](https://paperswithcode.com/dataset/utkface)

### AgeDB Dataset

AgeDB contains 16,488 images of various famous people, such as actors/actresses, writers, scientists, politicians, etc. Every image is annotated with respect to the identity, age, and gender attribute. There exist a total of 568 distinct subjects. The average number of images per subject is 29. The minimum and maximum age is 1 and 101, respectively. The average age range for each subject is 50.3 years.

- **Links to AgeDB**:
  - [Source](https://ibug.doc.ic.ac.uk/resources/agedb/)
  - [Kaggle](https://www.kaggle.com/datasets/nitingandhi/agedb-database?select=AgeDB)
  - [Papers with Code](https://paperswithcode.com/dataset/agedb)

### IMDB-WIKI Dataset

The IMDB-WIKI dataset is one of the largest publicly available datasets of face images with age and gender labels. It contains more than 500,000 images. This dataset is primarily used for pre-training models rather than for final evaluation.

- **Links to IMDB-WIKI**:
  - [IMDB-WIKI Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

To use this dataset, download `imdb_crop.tar` and `wiki_crop.tar`, and place them in `./data/imdb_crop` and `./data/wiki_crop`, respectively.

### Adience Dataset

The Adience dataset contains 26,580 photos across 2,284 subjects with a binary gender label and one label from eight different age groups, partitioned into five splits. It is used as a benchmark. You can find the state-of-the-art models [here](https://paperswithcode.com/sota/age-and-gender-classification-on-adience-age) for age and [here](https://paperswithcode.com/sota/age-and-gender-classification-on-adience) for gender, respectively.

- **Links to Adience**:
  - [Adience Dataset](https://paperswithcode.com/dataset/adience)
  - [Kaggle](https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification)

## Data Pre-processing

The data pre-processing steps involve loading the datasets, augmenting the images, and creating data generators for training and testing. The `load_data` function loads the UTKFace, AgeDB, WIKI, and IMDB datasets, extracting age and gender information from the filenames and storing them in a pandas DataFrame. Data generators are created using TensorFlow's `ImageDataGenerator` class, which applies various augmentations to the training images to improve the model's robustness. The `create_generators` function sets up these generators for both training and testing datasets. A custom data generator reshapes the gender labels to match the expected input shape for the model, ensuring that the data is correctly formatted for training. The `dataset_from_generator` function creates a TensorFlow dataset from the custom generator, which can be used for training and evaluation.

By following these pre-processing steps, the data is prepared for training the model, ensuring that the images are appropriately augmented and the labels are correctly formatted. The project uses models like ArcFace to extract features and embeddings from the images, which are then used for age and gender prediction.

## Models

The project uses deep learning models to predict age and gender from facial images. The model architectures are based on the VGG16 and ResNet networks, which are pre-trained on the ImageNet dataset and used for feature extraction. The models are fine-tuned for the specific tasks of age and gender prediction.

### VGG16 Model

The VGG16 model is a convolutional neural network that is 16 layers deep. It is widely used for image classification tasks and is known for its simplicity and effectiveness. In this project, VGG16 is used as a feature extractor (backbone), leveraging its pre-trained weights on the ImageNet dataset to provide a strong foundation for age and gender prediction.

#### Model Architecture

1. **Feature Extraction**: The [VGG16](https://arxiv.org/abs/1409.1556) model is used as the base for feature extraction. The top layers of VGG16 are removed, and the remaining layers are frozen to prevent them from being updated during training.
2. **Flattening**: The output of the VGG16 model is flattened to create a single long feature vector.
3. **Dense Layers**: Two separate dense layers are added for age and gender prediction. Each dense layer is followed by additional dense layers to further process the features.
4. **Output Layers**: 
   - The age prediction output layer uses a linear activation function.
   - The gender prediction output layer uses a sigmoid activation function.

#### Model Compilation

The model is compiled with the following configurations:
- **Optimizer**: AdamW
- **Loss Functions**: 
  - Age: Mean Squared Error (MSE)
  - Gender: Binary Cross-Entropy
- **Metrics**: 
  - Age: Mean Absolute Error (MAE)
  - Gender: Accuracy
- **Links to VGG16**:
  - [keras](https://keras.io/api/applications/vgg/)
  - [Papers with Code](https://paperswithcode.com/method/vgg)

### ResNet Model

The ResNet model is a convolutional neural network that is 50 layers deep. It is known for its residual learning framework, which allows it to train very deep networks without the vanishing gradient problem. In this project, ResNet50 is used as a feature extractor (backbone), leveraging its pre-trained weights on the ImageNet dataset to provide a strong foundation for age and gender prediction.

#### Model Architecture

1. **Feature Extraction**: The [ResNet50](https://arxiv.org/abs/1512.03385) model is used as the base for feature extraction. The top layers of ResNet50 are removed, and the remaining layers are frozen to prevent them from being updated during training.
2. **Flattening**: The output of the ResNet50 model is flattened to create a single long feature vector.
3. **Dense Layers**: Two separate dense layers are added for age and gender prediction. Each dense layer is followed by additional dense layers to further process the features.
4. **Output Layers**: 
   - The age prediction output layer uses a linear activation function.
   - The gender prediction output layer uses a sigmoid activation function.

#### Model Compilation

The model is compiled with the following configurations:
- **Optimizer**: AdamW
- **Loss Functions**: 
  - Age: Mean Squared Error (MSE)
  - Gender: Binary Cross-Entropy
- **Metrics**: 
  - Age: Mean Absolute Error (MAE)
  - Gender: Accuracy

- **Links to ResNet50**:
  - [keras](https://keras.io/api/applications/resnet/)
  - [Papers with Code](https://paperswithcode.com/method/resnet)

