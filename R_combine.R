library(keras)

# load the fashion_mnist dataset
fashion_mnist_train_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_train.csv')
fashion_mnist_test_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_test.csv')

# show some examples of the pictures
par(mfrow = c(2,2))
image(matrix(as.numeric(fashion_mnist_train_dat[43,2:785]),28,28), main = as.numeric(fashion_mnist_train_dat[43,1]))
image(matrix(as.numeric(fashion_mnist_train_dat[143,2:785]),28,28), main = as.numeric(fashion_mnist_train_dat[143,1]))
image(matrix(as.numeric(fashion_mnist_train_dat[437,2:785]),28,28), main = as.numeric(fashion_mnist_train_dat[437,1]))
image(matrix(as.numeric(fashion_mnist_train_dat[5413,2:785]),28,28), main = as.numeric(fashion_mnist_train_dat[5413,1]))

par(mfrow = c(1,3))
image(matrix(as.numeric(fashion_mnist_train_dat[75,2:785]),28,28), main = as.numeric(fashion_mnist_train_dat[75,1]))
image(matrix(as.numeric(fashion_mnist_train_dat[78,2:785]),28,28), main = as.numeric(fashion_mnist_train_dat[78,1]))
image(matrix(as.numeric(fashion_mnist_train_dat[86,2:785]),28,28), main = as.numeric(fashion_mnist_train_dat[86,1]))

fashion_mnist_train_images <- array(as.numeric(as.matrix(fashion_mnist_train_dat[,2:785],nrow=60000,ncol=784)), dim=c(60000,28,28))
fashion_mnist_train_labels <- array(as.integer(as.matrix(fashion_mnist_train_dat[,1],nrow=60000,ncol=1)))
fashion_mnist_test_images <- array(as.numeric(as.matrix(fashion_mnist_test_dat[,2:785],nrow=10000,ncol=784)), dim=c(10000,28,28))
fashion_mnist_test_labels <- array(as.integer(as.matrix(fashion_mnist_test_dat[,1],nrow=10000,ncol=1)))

rm(fashion_mnist_train_dat, fashion_mnist_test_dat)

# format the labels such that it is readable by keras
fashion_mnist_train_labels <- to_categorical(fashion_mnist_train_labels)
fashion_mnist_test_labels <- to_categorical(fashion_mnist_test_labels)

# standardize the datasets to feed the neural network
fashion_mnist_train_images <- array_reshape(fashion_mnist_train_images, c(60000, 784))
fashion_mnist_train_images <- fashion_mnist_train_images / 255
fashion_mnist_test_images <- array_reshape(fashion_mnist_test_images, c(10000, 784))
fashion_mnist_test_images <- fashion_mnist_test_images / 255

# build the neural network structure with regularization
fashion_mnist_model_NN <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")
fashion_mnist_model_NN

# define the optimizer, loss function and the performance metric
fashion_mnist_model_NN %>% 
  compile(optimizer = "rmsprop",
          loss = "categorical_crossentropy",
          metrics = c("accuracy"))

# train the model and predict on the validation dataset
history_fmnist_NN <- fashion_mnist_model_NN %>% fit(fashion_mnist_train_images, 
                                                    fashion_mnist_train_labels,
                                                    validation_data = list(fashion_mnist_test_images, 
                                                                           fashion_mnist_test_labels),
                                                    epochs = 50, batch_size = 128)

# plot the performance vs epoch
plot(history_fmnist_NN) + ggtitle("fashion_mnist")

# load the mnist dataset
mnist <- dataset_mnist()

mnist_train_images <- mnist$train$x
mnist_train_labels <- mnist$train$y
mnist_test_images <- mnist$test$x
mnist_test_labels <- mnist$test$y

mnist_train_labels <- to_categorical(mnist_train_labels)
mnist_test_labels <- to_categorical(mnist_test_labels)

# standardize
mnist_train_images <- array_reshape(mnist_train_images, c(60000, 784))
mnist_train_images <- mnist_train_images / 255
mnist_test_images <- array_reshape(mnist_test_images, c(10000, 784))
mnist_test_images <- mnist_test_images / 255

rm(mnist)

# use mnist dataset to train the fashion_mnist neural network
history_mnist_NN <- fashion_mnist_model_NN %>% fit(mnist_train_images, 
                                                   mnist_train_labels,
                                                   validation_data = list(mnist_test_images, 
                                                                          mnist_test_labels),
                                                   epochs = 50, batch_size = 128)

# plot the performance vs epoch
plot(history_mnist_NN) + ggtitle("mnist")

# CNN_1: CNN without regularization
model_cnn1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), padding = "same", activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu", input_shape = c(28,28,1)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model_cnn1

model_cnn1 %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))

history_cnn1 <- model_cnn1 %>% fit(fashion_mnist_train_images, 
                                   fashion_mnist_train_labels, 
                                   validation_data = list(fashion_mnist_test_images, 
                                                          fashion_mnist_test_labels), 
                                   epochs = 15, batch_size = 64)
plot(history_cnn1) + ggtitle("CNN_1")
model_cnn1 %>% save_model_hdf5("fashion_mnist_CNN_1.h5")

## CNN_2: CNN_1 with 0.5 dropout rate after flattening

model_cnn2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), padding = "same", activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu", input_shape = c(28,28,1)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model_cnn2

model_cnn2 %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))

history_cnn2 <- model_cnn2 %>% fit(fashion_mnist_train_images, 
                                   fashion_mnist_train_labels, 
                                   validation_data = list(fashion_mnist_test_images, 
                                                          fashion_mnist_test_labels), 
                                   epochs = 15, batch_size = 64)
plot(history_cnn2) + ggtitle("CNN_2")
model_cnn2 %>% save_model_hdf5("fashion_mnist_CNN_2.h5")

## CNN_3: CNN_2 with L2 regularization

model_cnn3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), padding = "same", activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu", input_shape = c(28,28,1)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.001), activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model_cnn3

model_cnn3 %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))

history_cnn3 <- model_cnn3 %>% fit(fashion_mnist_train_images, 
                                   fashion_mnist_train_labels, 
                                   validation_data = list(fashion_mnist_test_images, 
                                                          fashion_mnist_test_labels), 
                                   epochs = 50, batch_size = 64)
plot(history_cnn3) + ggtitle("CNN_3")
model_cnn3 %>% save_model_hdf5("fashion_mnist_CNN_3.h5")