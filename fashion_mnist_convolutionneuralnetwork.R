library(keras)

train_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_train.csv')
test_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_test.csv')

train_images <- array(as.numeric(as.matrix(train_dat[,2:785], nrow = 60000, ncol = 784)), dim = c(60000,28,28))
train_labels <- array(as.integer(as.matrix(train_dat[,1], nrow = 60000, ncol = 1)))
test_images <- array(as.numeric(as.matrix(test_dat[,2:785], nrow = 10000, ncol = 784)), dim = c(10000,28,28))
test_labels <- array(as.integer(as.matrix(test_dat[,1], nrow = 10000, ncol = 1)))

rm(train_dat, test_dat)

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255

## CNN_1: CNN without regularization

model_cnn1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model_cnn1

model_cnn1 %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))

history_cnn1 <- model_cnn1 %>% fit(train_images, train_labels, 
                                   validation_data = list(test_images, test_labels), 
                                   epochs = 15, batch_size = 64)
plot(history_cnn1)
model_cnn1 %>% save_model_hdf5("fashion_mnist_CNN_1.h5")

## CNN_2: CNN_1 with 0.5 dropout rate after flattening

model_cnn2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model_cnn2

model_cnn2 %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))

history_cnn2 <- model_cnn2 %>% fit(train_images, train_labels, 
                                   validation_data = list(test_images, test_labels), 
                                   epochs = 15, batch_size = 64)
plot(history_cnn2)
model_cnn2 %>% save_model_hdf5("fashion_mnist_CNN_2.h5")

## CNN_3: CNN_2 with L2 regularization

model_cnn3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64, kernel_regularizer = regularizer_l2(0.001), activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model_cnn3

model_cnn3 %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))

history_cnn3 <- model_cnn3 %>% fit(train_images, train_labels, 
                                   validation_data = list(test_images, test_labels), 
                                   epochs = 50, batch_size = 64)
plot(history_cnn3)
model_cnn3 %>% save_model_hdf5("fashion_mnist_CNN_3.h5")