library(keras)

train_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_train.csv')
test_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_test.csv')

train_images <- array(as.numeric(as.matrix(train_dat[,2:785],nrow=60000,ncol=784)), dim=c(60000,28,28))
train_labels <- array(as.integer(as.matrix(train_dat[,1],nrow=60000,ncol=1)))
test_images <- array(as.numeric(as.matrix(test_dat[,2:785],nrow=10000,ncol=784)), dim=c(10000,28,28))
test_labels <- array(as.integer(as.matrix(test_dat[,1],nrow=10000,ncol=1)))

train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255

network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(512)) %>%
  layer_dense(units = 10, activation = "softmax", input_shape = c(256))

network %>% 
  compile(optimizer = "rmsprop",
          loss = "categorical_crossentropy",
          metrics = c("accuracy"))

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

network %>% 
  fit(train_images, train_labels, epochs = 5, batch_size = 128)

metrics <- network %>% 
  evaluate(test_images, test_labels)
metrics
