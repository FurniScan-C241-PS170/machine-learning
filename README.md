# FurniScan - Machine Learning

## Prerequisites

 install:

- **TensorFlow 2.x**
- **Keras (included with TensorFlow 2.x)**
  
You can install TensorFlow using pip:
```
pip install tensorflow

```
## steps to create a model

**1. Import Libraries:**

- ImageDataGenerator from tensorflow.keras.preprocessing.image is used for data preprocessing and augmentation.

**2. Set Parameters:**

- img_height and img_width define the dimensions to resize all images to.
- batch_size specifies the number of images to be yielded from the generator per batch.

**3. Create Training Data Generator:**

- train_datagen is configured with various augmentation techniques such as rescaling, rotation, shifting, shearing, zooming, and flipping.

**4. Create Validation Data Generator:**

- val_datagen is configured to only rescale the images.

**5. Load Training Data:**

- train_generator loads training data from the train directory, applies augmentations, and prepares the data in batches.

**6. Load Validation Data:**

- validation_generator loads validation data from the validation directory, rescaling the images without further augmentations.

**7. Training the Model**
