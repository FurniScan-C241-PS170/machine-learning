# FurniScan - Machine Learning
![download](https://github.com/FurniScan-C241-PS170/machine-learning/assets/147247858/7c701001-c139-4fb3-947c-cc271c732890)

We utilize **Google Colab** for our Python Notebook to create a Machine Learning model. Colab is a hosted Jupyter Notebook service that requires no setup to use and provides free access to compute resources, including GPU and TPU.


## Prerequisites

 install:

- **Python 3.6 or higher**
- **TensorFlow 2.x**
- **Keras (included with TensorFlow 2.x)**
  
You can install TensorFlow using pip:
```
pip install tensorflow

```

## Technology We Used

![download](https://github.com/FurniScan-C241-PS170/machine-learning/assets/147247858/0f3c1e84-261c-47d4-8d3c-161f5e675797)
![download](https://github.com/FurniScan-C241-PS170/machine-learning/assets/147247858/291d105d-6dae-4780-9385-d58a1b6bf44d)

## Steps to Create a Model

**1. Import Libraries:**

- `ImageDataGenerator` from `tensorflow.keras.preprocessing.image` is used for data preprocessing and augmentation.

**2. Set Parameters:**

- `img_height` and `img_width` define the dimensions to resize all images to.
- `batch_size` specifies the number of images to be yielded from the generator per batch.

**3. Create Training Data Generator:**

- `train_datagen` is configured with various augmentation techniques such as rescaling, rotation, shifting, shearing, zooming, and flipping.

**4. Create Validation Data Generator:**

- `val_datagen` is configured to only rescale the images.

**5. Load Training Data:**

- `train_generator` loads training data from the train directory, applies augmentations, and prepares the data in batches.

**6. Load Validation Data:**

- `validation_generator` loads validation data from the validation directory, rescaling the images without further augmentations.

**7. Training the Model**
