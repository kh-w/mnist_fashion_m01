# Objective
- Compare the performances of artificial neural network and convolution neural networks on the fashion version of mnist dataset 
- Seek the best tuned convolution neural network to predict category of fashion_mnist images

# Dataset
fashion_mnist is a fashion version of mnist (the famous toy dataset for image recognition) and it is obtainable at [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist). It is a dataset of 60000 (train) + 10000 (validate) = 70000 (total) Zalando's article images with each image having the same dimension (28 x 28 pixels) and all images are categorized into 10 categories, similar to mnist (digit 0-9). The 10 categories are listed below. 

Below are 4 sample images from the dataset:

<img src="/plot_1.png" align="left" height="408" width="408">

| levels  | category |
| ------------- | ------------- |
|0|T-shirt/top|
|1|Trouser|
|2|Pullover|
|3|Dress|
|4|Coat|
|5|Sandal|
|6|Shirt|
|7|Sneaker|
|8|Bag|
|9|Ankle boot|

According to Zalando Research, they seek to replace mnist dataset by the fashion_mnist dataset for those who want to benchmark their image recognition algorithms, for instance, CNN. Since most CNNs work almost equally well on traditional mnist dataset, this replacement is beneficial to those who want to test out their CNNs such that it is easier to distinguish good or bad models.

### Compare fashion_mnist to mnist
To compare the predictability of mnist and fashion_mnist, a simple artificial neural network is useful. Keras library (<a href="https://cran.r-project.org/web/packages/keras/index.html">link</a>) is needed throughout the whole project.
Using the following regularized artificial neural network:

<table>
  <tr>
    <td colspan = "2"><img src="/fashion_mnist_model_NN.png"></td>
  <tr>
  <tr>
    <td><img src="/fashion_mnist_NN.png"></td>
    <td><img src="/mnist_NN.png"></td>
  </tr>
</table>

The accuracy is around 89% on the fashion_mnist dataset but the accuracy is around 98% on the mnist dataset. The performances differ because of the amount of noise, the variety within a single category and the similarity between two or more categories. The below comparison shows how subtle the differences could be.

<img src="/similar_images.png">

Since artificial neural network "observes" the input as a whole picture, subtle differences are more difficult to detect. In this case, a convolution neural network is more appropriate because the key difference between neural network and convolution neural network (CNN) is that CNN decompose a picture into translational invariable local patterns.

# Convolution Neural Network
This section describes the baseline Convolution Neural Network for this project, and the next section will be the fine tuning base on this model.

A CNN model tries to recognize translation invariant patterns of the images, and use these patterns to identify the category for a single image. 
The process would be the following:

1. Feed the image to the first layer
2. The first layer produces response maps by some _filters_ (containing the parameters)
3. The second layer reads the output of the first layer and perform similar processes
4. Keep going until the last layer
5. Flatten the output of the last layer
6. Feed the flattened output to an artificial nerual network and predict
7. Minimize the loss by updating the parameters in favor to reduce loss

The input of the CNN model would be a _tensor_ with shape [1:10000, 1:28, 1:28, 1] which is a collection of 10000 28 x 28 images with 1 color scale (grayscale). The first layer of the model is a Conv2D. The process is to "slide" _filters_ through the input image and output _channels_. 

layer_conv_2d(filters = 32, 
                kernel_size = c(3,3), 
                padding = "same", 
                activation = "relu", input_shape = c(28,28,1))

_tensor_: In Keras, tensor is the standard input/output format of a layer in a nerual network
_kernel_: A kernel in CNN is usually a 3x3 or 4x4 matrix containing the weights
_filter_: Filter is a stack of n kernels, where n is the number of _channels_ of the image/input
_channel_: A channel is a _response map_, the result of "sliding" the filter through the image/input
_response map_: A response map is a sum of matrices, each matrix is the dot product of the kernel and part of the image/input

The number of filters in a layer dictates the number of channels in the layer's output. Therefore, the first layer uses 32 filters to output a 32 channels tensor, i.e. Output Shape (None, 28, 28, 32).
