# Objective
- Understand CNN (Convolution Neural Network)
- Compare the performances of simple neural networks and convolutional neural networks on the fashion version of mnist dataset 
- Seek the best tuned convolution neural network.

# Dataset
fashion_mnist is a fashion version of mnist (the famous toy dataset for image recognition) and it is obtainable at [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist). It is a dataset of 60000 (train) + 10000 (validate) = 70000 (total) Zalando's article images with each image having the same dimension (28 * 28 pixels) and all images are categorized into 10 categories, similar to mnist (digit 0-9). The 10 categories are listed below. 

Below are 4 sample images from the dataset:

<img src="/plot_1.png" align="left" height="410" width="410">

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

According to Zalando Research, they seek to replace mnist dataset by the fashion_mnist dataset for those who want to benchmark their image recognition algorithms. Since most CNNs work almost equally well on traditional mnist dataset, this replacement is beneficial to those who want to test out their CNN.

# Compare fashion_mnist to mnist
To compare the predictability of mnist and fashion_mnist, a simple neural network is useful. Using the following regularized neural network:
<img src="/fashion_mnist_model_NN.png" align="left">
