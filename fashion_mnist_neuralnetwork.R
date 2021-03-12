library(keras)

train_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_train.csv')
test_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_test.csv')

train_images <- array(as.numeric(as.matrix(train_dat[,2:785],nrow=60000,ncol=784)), dim=c(60000,28,28))
train_labels <- array(as.integer(as.matrix(train_dat[,1],nrow=60000,ncol=1)))
test_images <- array(as.numeric(as.matrix(test_dat[,2:785],nrow=10000,ncol=784)), dim=c(10000,28,28))
test_labels <- array(as.integer(as.matrix(test_dat[,1],nrow=10000,ncol=1)))

rm(train_dat, test_dat)

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

train_images <- array_reshape(train_images, c(60000, 784))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 784))
test_images <- test_images / 255

# NN_1: neural network without regularization

model_nn1 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(784)) %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(512)) %>%
  layer_dense(units = 10, activation = "softmax", input_shape = c(256))

model_nn1 %>% 
  compile(optimizer = "rmsprop",
          loss = "categorical_crossentropy",
          metrics = c("accuracy"))

history_nn1 <- model_nn1 %>% fit(train_images, train_labels,
                                 validation_data = list(test_images, test_labels),
                                 epochs = 25, batch_size = 128)
plot(history_nn1)
model_nn1 %>% save_model_hdf5("fashion_mnist_NN_1.h5")

# NN_2: NN_1 with 0.5 dropout rate before output layer

model_nn2 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(784)) %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(512)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax", input_shape = c(256))

model_nn2 %>% 
  compile(optimizer = "rmsprop",
          loss = "categorical_crossentropy",
          metrics = c("accuracy"))

history_nn2 <- model_nn2 %>% fit(train_images, train_labels,
                                 validation_data = list(test_images, test_labels),
                                 epochs = 25, batch_size = 128)
plot(history_nn2)
model_nn2 %>% save_model_hdf5("fashion_mnist_NN_2.h5")

# NN_3: NN_2 with L2 regularization

model_nn3 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", kernel_regularizer = regularizer_l2(0.001), input_shape = c(784)) %>%
  layer_dense(units = 256, activation = "relu", kernel_regularizer = regularizer_l2(0.001), input_shape = c(512)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax", input_shape = c(256))

model_nn3 %>% 
  compile(optimizer = "rmsprop",
          loss = "categorical_crossentropy",
          metrics = c("accuracy"))

history_nn3 <- model_nn3 %>% fit(train_images, train_labels,
                                 validation_data = list(test_images, test_labels),
                                 epochs = 25, batch_size = 128)
plot(history_nn3)
model_nn3 %>% save_model_hdf5("fashion_mnist_NN_3.h5")