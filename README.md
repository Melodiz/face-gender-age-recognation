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

## Data Pre-processing

The data pre-processing steps involve loading the datasets, augmenting the images, and creating data generators for training and testing. The `load_data` function loads the UTKFace and AgeDB datasets, extracting age and gender information from the filenames and storing them in a pandas DataFrame. Data generators are created using TensorFlow's `ImageDataGenerator` class, which applies various augmentations to the training images to improve the model's robustness. The `create_generators` function sets up these generators for both training and testing datasets. A custom data generator reshapes the gender labels to match the expected input shape for the model, ensuring that the data is correctly formatted for training. The `dataset_from_generator` function creates a TensorFlow dataset from the custom generator, which can be used for training and evaluation.

By following these pre-processing steps, the data is prepared for training the model, ensuring that the images are appropriately augmented and the labels are correctly formatted. The project uses models like ArcFace to extract features and embeddings from the images, which are then used for age and gender prediction.

## Model

The project uses a deep learning model to predict age and gender from facial images. The model architecture is based on the VGG16 network, which is pre-trained on the ImageNet dataset and used for feature extraction. The VGG16 model is fine-tuned for the specific tasks of age and gender prediction.

### Model Architecture

1. **Feature Extraction**: The [VGG16](https://arxiv.org/abs/1409.1556) model is used as the base for feature extraction. The top layers of VGG16 are removed, and the remaining layers are frozen to prevent them from being updated during training.
2. **Flattening**: The output of the VGG16 model is flattened to create a single long feature vector.
3. **Dense Layers**: Two separate dense layers are added for age and gender prediction. Each dense layer is followed by additional dense layers to further process the features.
4. **Output Layers**: 
   - The age prediction output layer uses a linear activation function.
   - The gender prediction output layer uses a sigmoid activation function.

### Model Compilation

The model is compiled with the following configurations:
- **Optimizer**: AdamW
- **Loss Functions**: 
  - Age: Mean Squared Error (MSE)
  - Gender: Binary Cross-Entropy
- **Metrics**: 
  - Age: Mean Absolute Error (MAE)
  - Gender: Accuracy

### Training and Evaluation

The model is trained using the UTKFace and AgeDB datasets, with data augmentation applied to improve generalization. The training process includes:
- Splitting the dataset into training and testing sets.
- Creating data generators for efficient data loading.
- Using TensorBoard for monitoring training progress.

The model is evaluated on a separate test set to measure its performance in terms of loss, age prediction mean squared error, and gender prediction accuracy.

### VGG16 Model

The VGG16 model is a convolutional neural network that is 16 layers deep. It is widely used for image classification tasks and is known for its simplicity and effectiveness. In this project, VGG16 is used as a feature extractor, leveraging its pre-trained weights on the ImageNet dataset to provide a strong foundation for age and gender prediction.

- **Links to VGG16**:
  - [keras](https://keras.io/api/applications/vgg/)
  - [Papers with Code](https://paperswithcode.com/method/vgg)

### Evaluation results

The application for using the university supercomputer is being processed. 
### Estimated Complexity

- **Total Parameters**: 34,116,418 (130.14 MB)
- **Trainable Parameters**: 19,401,730 (74.01 MB)
- **Non-trainable Parameters**: 14,714,688 (56.13 MB)
- **Training Size**: 32,157 images
- **Test Size**: 8,039 images

The model uses approximately 6.8e7 FLOPs for each forward pass. It was trained on 32,157 data points, resulting in a total of approximately 6.5e12 FLOPs during training. Therefore, each epoch requires around 6.5 TFlops.

As a result, we can classify this model as lightweight.