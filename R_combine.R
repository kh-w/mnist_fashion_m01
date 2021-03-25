library(keras)
library(dplyr)
library(ggplot2)

############################################################################################################
#######################################--- import dataset ---###############################################

# load the fashion_mnist dataset
mnist_f_train_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_train.csv')
mnist_f_test_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_test.csv')

# show some examples of the pictures
par(mfrow = c(2,2))
image(matrix(as.numeric(mnist_f_train_dat[43,2:785]),28,28), main = as.numeric(mnist_f_train_dat[43,1]))
image(matrix(as.numeric(mnist_f_train_dat[143,2:785]),28,28), main = as.numeric(mnist_f_train_dat[143,1]))
image(matrix(as.numeric(mnist_f_train_dat[437,2:785]),28,28), main = as.numeric(mnist_f_train_dat[437,1]))
image(matrix(as.numeric(mnist_f_train_dat[5413,2:785]),28,28), main = as.numeric(mnist_f_train_dat[5413,1]))
par(mfrow = c(1,3))
image(matrix(as.numeric(mnist_f_train_dat[75,2:785]),28,28), main = as.numeric(mnist_f_train_dat[75,1]))
image(matrix(as.numeric(mnist_f_train_dat[78,2:785]),28,28), main = as.numeric(mnist_f_train_dat[78,1]))
image(matrix(as.numeric(mnist_f_train_dat[86,2:785]),28,28), main = as.numeric(mnist_f_train_dat[86,1]))

# reshape the dataset
mnist_f_train_images <- array(as.numeric(as.matrix(mnist_f_train_dat[,2:785],
                                                         nrow=60000,
                                                         ncol=784)), 
                                    dim=c(60000,28,28,1))
mnist_f_train_labels <- array(as.integer(as.matrix(mnist_f_train_dat[,1],
                                                         nrow=60000,
                                                         ncol=1)))
mnist_f_test_images <- array(as.numeric(as.matrix(mnist_f_test_dat[,2:785],
                                                        nrow=10000,
                                                        ncol=784)), 
                                   dim=c(10000,28,28,1))
mnist_f_test_labels <- array(as.integer(as.matrix(mnist_f_test_dat[,1],
                                                        nrow=10000,
                                                        ncol=1)))

# format the labels such that it is readable by keras
mnist_f_train_labels <- to_categorical(mnist_f_train_labels)
mnist_f_test_labels <- to_categorical(mnist_f_test_labels)

# standardize the datasets to feed the neural network
mnist_f_train_images_nn <- array_reshape(mnist_f_train_images, c(60000, 784))
mnist_f_train_images_nn <- mnist_f_train_images_nn / 255
mnist_f_test_images_nn <- array_reshape(mnist_f_test_images, c(10000, 784))
mnist_f_test_images_nn <- mnist_f_test_images_nn / 255

# load the mnist dataset
mnist <- dataset_mnist()
mnist_train_images <- mnist$train$x
mnist_train_labels <- mnist$train$y
mnist_test_images <- mnist$test$x
mnist_test_labels <- mnist$test$y
mnist_train_labels <- to_categorical(mnist_train_labels)
mnist_test_labels <- to_categorical(mnist_test_labels)

# reshape and standardize
mnist_train_images_nn <- array_reshape(mnist_train_images, c(60000, 784))
mnist_train_images_nn <- mnist_train_images_nn / 255
mnist_test_images_nn <- array_reshape(mnist_test_images, c(10000, 784))
mnist_test_images_nn <- mnist_test_images_nn / 255

# reshape and standardize mnist for CNN
mnist_train_images <- array_reshape(mnist_train_images, c(60000, 28, 28, 1))
mnist_train_images <- mnist_train_images / 255
mnist_test_images <- array_reshape(mnist_test_images, c(10000, 28, 28, 1))
mnist_test_images <- mnist_test_images / 255

############################################################################################################
########################################----- ANN -----#####################################################

# build the neural network structure with regularization
model_NN <- keras_model_sequential() %>%
  layer_dense(units = 512, kernel_regularizer = regularizer_l2(0.001), activation = "relu", input_shape = c(784)) %>%
  layer_dense(units = 256, kernel_regularizer = regularizer_l2(0.001), activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")

# define the optimizer, loss function and the performance metric
model_NN %>% 
  compile(optimizer = "rmsprop",
          loss = "categorical_crossentropy",
          metrics = c("accuracy"))

# train the model and predict on the validation dataset
history_mnist_f_NN <- model_NN %>% fit(mnist_f_train_images_nn, 
                                       mnist_f_train_labels,
                                       validation_data = list(mnist_f_test_images_nn, 
                                                              mnist_f_test_labels),
                                       epochs = 20, batch_size = 128)

# plot the performance vs epoch
plot_history_mnist_f_NN <- plot(history_mnist_f_NN) + ggtitle("Artificial Neural Network trained by fashion_mnist")

# save the model
model_mnist_f_NN <- model_NN
model_mnist_f_NN %>% save_model_hdf5("model_mnist_f_NN.h5")
load_model_hdf5("model_mnist_f_NN.h5")

# use mnist dataset to train the fashion_mnist neural network
model_NN <- keras_model_sequential() %>%
  layer_dense(units = 512, kernel_regularizer = regularizer_l2(0.001), activation = "relu", input_shape = c(784)) %>%
  layer_dense(units = 256, kernel_regularizer = regularizer_l2(0.001), activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")
model_NN %>% 
  compile(optimizer = "rmsprop",
          loss = "categorical_crossentropy",
          metrics = c("accuracy"))
history_mnist_NN <- model_NN %>% fit(mnist_train_images_nn, 
                                     mnist_train_labels,
                                     validation_data = list(mnist_test_images_nn, 
                                                            mnist_test_labels),
                                     epochs = 20, batch_size = 128)

# plot the performance vs epoch
plot_history_mnist_NN <- plot(history_mnist_NN) + ggtitle("Artificial Neural Network trained by mnist")

# save the model
model_mnist_NN <- model_NN
model_mnist_NN %>% save_model_hdf5("model_mnist_NN.h5")
load_model_hdf5("model_mnist_NN.h5")

rm(model_NN)

############################################################################################################

# mnist training dataset for ANN = mnist_train_images_nn
# mnist testing dataset for ANN = mnist_test_images_nn

# mnist training dataset for CNN = mnist_train_images
# mnist testing dataset for CNN = mnist_test_images

# fashion mnist training dataset for ANN = mnist_f_train_images_nn
# fashion mnist testing dataset for ANN = mnist_f_test_images_nn

# fashion mnist training dataset for CNN = mnist_f_train_images
# fashion mnist testing dataset for CNN = mnist_f_test_images

############################################################################################################
########################################----- CNN -----#####################################################

# CNN without regularization
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3), 
                activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3), 
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3), 
                activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

# compile the model
model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))

# train using fashion_mnist
history_cnn_no_reg <- model %>% fit(mnist_f_train_images, 
                                    mnist_f_train_labels, 
                                    validation_data = list(mnist_f_test_images, 
                                                           mnist_f_test_labels), 
                                    epochs = 20, 
                                    batch_size = 64)

# plot the loss and accuracy against epochs
plot(history_cnn_no_reg) + ggtitle("CNN without regularization")

# save the model
model_CNN_no_reg <- model
model_CNN_no_reg %>% save_model_hdf5("mnist_f_CNN_no_reg.h5")
load_model_hdf5("mnist_f_CNN_no_reg.h5")

## implement 0.5 dropout rate after flattening
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3), 
                activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3), 
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3), 
                activation = "relu") %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

# compile the model
model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))

# train using fashion_mnist
history_cnn_dropout <- model%>% fit(mnist_f_train_images, 
                                    mnist_f_train_labels, 
                                    validation_data = list(mnist_f_test_images, 
                                                           mnist_f_test_labels),
                                    epochs = 20, 
                                    batch_size = 64)

# plot the loss and accuracy against epochs
plot(history_cnn_dropout) + ggtitle("CNN with 0.5 dropout rate")

# save the model
model_CNN_dropout <- model
model_CNN_dropout %>% save_model_hdf5("mnist_f_CNN_dropout.h5")
load_model_hdf5("mnist_f_CNN_dropout.h5")

## implement L2 regularization
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3), 
                activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3), 
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3), 
                activation = "relu") %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, kernel_regularizer = regularizer_l2(0.001), activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")
model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))

# train using fashion_mnist
history_cnn_dropout_L2 <- model %>% fit(mnist_f_train_images, 
                                        mnist_f_train_labels, 
                                        validation_data = list(mnist_f_test_images, 
                                                               mnist_f_test_labels), 
                                        epochs = 20, batch_size = 64)

# plot the loss and accuracy against epochs
plot(history_cnn_dropout_L2) + ggtitle("CNN_with_dropout_and_L2_reg")

# save the model
model_CNN_dropout_L2 <- model
model_CNN_dropout_L2 %>% save_model_hdf5("mnist_f_CNN_dropout_L2.h5")
load_model_hdf5("mnist_f_CNN_dropout_L2.h5")

# train using mnist
history_cnn_dropout_L2_mnist <- model %>% fit(mnist_train_images,
                                              mnist_train_labels,
                                              validation_data = list(mnist_test_images,
                                                                     mnist_test_labels),
                                              epochs = 20, batch_size = 64)

# plot the loss and accuracy against epochs
plot(history_cnn_dropout_L2_mnist) + ggtitle("CNN_with_dropout_and_L2_reg_mnist")

rm(model)

############################################################################################################
########################################----- CNN tuning -----##############################################

# a function to fill the comparison table
get_result <- function(i,input){
  conv_layer_1st <- input[1]
  conv_layer_2nd <- input[2]
  conv_layer_3rd <- input[3]
  kernel <- input[4]
  layer_dense <- input[5]
  epoch <- input[6]
  tic <- Sys.time()
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = conv_layer_1st, 
                  kernel_size = c(kernel,kernel), 
                  activation = "relu", input_shape = c(28,28,1)) %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_conv_2d(filters = conv_layer_2nd, 
                  kernel_size = c(kernel,kernel), 
                  activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_conv_2d(filters = conv_layer_3rd, 
                  kernel_size = c(kernel,kernel), 
                  activation = "relu") %>%
    layer_flatten() %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = layer_dense, 
                kernel_regularizer = regularizer_l2(0.001), 
                activation = "relu") %>%
    layer_dense(units = 10, activation = "softmax")
  model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))
  history <- model %>% fit(mnist_f_train_images,
                           mnist_f_train_labels,
                           validation_data = list(mnist_f_test_images,
                                                  mnist_f_test_labels),
                           epochs = epoch, batch_size = 64)
  toc <- Sys.time()
  
  print(paste(i,"-",tail(history$metrics$val_accuracy,1)))
  
  # output = c(runtime (mins), loss, accuracy)
  c(toc-tic,tail(history$metrics$loss,1),tail(history$metrics$val_accuracy,1))
}
get_result_2layer <- function(i,input){
  conv_layer_1st <- input[1]
  conv_layer_2nd <- input[2]
  kernel <- input[3]
  layer_dense <- input[4]
  epoch <- input[5]
  tic <- Sys.time()
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = conv_layer_1st, 
                  kernel_size = c(kernel,kernel), 
                  activation = "relu", input_shape = c(28,28,1)) %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_conv_2d(filters = conv_layer_2nd, 
                  kernel_size = c(kernel,kernel), 
                  activation = "relu") %>%
    layer_flatten() %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = layer_dense, 
                kernel_regularizer = regularizer_l2(0.001), 
                activation = "relu") %>%
    layer_dense(units = 10, activation = "softmax")
  model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))
  history <- model %>% fit(mnist_f_train_images,
                           mnist_f_train_labels,
                           validation_data = list(mnist_f_test_images,
                                                  mnist_f_test_labels),
                           epochs = epoch, batch_size = 64)
  toc <- Sys.time()
  
  print(paste(i,"-",tail(history$metrics$val_accuracy,1)))
  
  # output = c(runtime (mins), loss, accuracy)
  c(toc-tic,tail(history$metrics$loss,1),tail(history$metrics$val_accuracy,1))
}
get_result_1layer <- function(i,input){
  conv_layer_1st <- input[1]
  kernel <- input[2]
  layer_dense <- input[3]
  epoch <- input[4]
  tic <- Sys.time()
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = conv_layer_1st, 
                  kernel_size = c(kernel,kernel), 
                  activation = "relu", input_shape = c(28,28,1)) %>%
    layer_flatten() %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = layer_dense, 
                kernel_regularizer = regularizer_l2(0.001), 
                activation = "relu") %>%
    layer_dense(units = 10, activation = "softmax")
  model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))
  history <- model %>% fit(mnist_f_train_images,
                           mnist_f_train_labels,
                           validation_data = list(mnist_f_test_images,
                                                  mnist_f_test_labels),
                           epochs = epoch, batch_size = 64)
  toc <- Sys.time()
  
  print(paste(i,"-",tail(history$metrics$val_accuracy,1)))
  
  # output = c(runtime (mins), loss, accuracy)
  c(toc-tic,tail(history$metrics$loss,1),tail(history$metrics$val_accuracy,1))
}

# tuneGrid: possible sets of parameters for the model
tuneGridResult <- expand.grid(layer1=c(16,32,64,128),
                              layer2=c(16,32,64,128),
                              layer3=c(16,32,64,128),
                              kernel=3,
                              layerdense=c(16,32,64,128),
                              epochs=20,
                              exclude=NA,
                              runtime_mins=NA,
                              loss=NA,
                              accuracy=NA)
tuneGridResult$exclude <- 
  tuneGridResult$layer2<tuneGridResult$layer1 | 
  tuneGridResult$layer3<tuneGridResult$layer2 |
  tuneGridResult$layerdense<tuneGridResult$layer3
tuneGridResult <- 
  tuneGridResult %>% 
  filter(exclude==FALSE) %>%
  select(-exclude)

# append model performance columns to the tuneGrid 
for (i in 1:nrow(tuneGridResult)){
  tuneGridResult[i,7:9] <- get_result(i, tuneGridResult[i,1:6])
}

# tuneGrid_2layer: possible sets of parameters for the model
tuneGridResult2layer <- expand.grid(layer1=c(16,32,64,128),
                                    layer2=c(16,32,64,128),
                                    kernel=3,
                                    layerdense=c(16,32,64,128),
                                    epochs=20,
                                    exclude=NA,
                                    runtime_mins=NA,
                                    loss=NA,
                                    accuracy=NA)
tuneGridResult2layer$exclude <- 
  tuneGridResult2layer$layer2<tuneGridResult2layer$layer1 | 
  tuneGridResult2layer$layerdense<tuneGridResult2layer$layer2
tuneGridResult2layer <- 
  tuneGridResult2layer %>% 
  filter(exclude==FALSE) %>%
  select(-exclude)

# append model performance columns to the tuneGrid 
for (i in 1:nrow(tuneGridResult2layer)){
  tuneGridResult2layer[i,6:8] <- get_result_2layer(i, tuneGridResult2layer[i,1:5])
}

# tuneGrid_1layer: possible sets of parameters for the model
tuneGridResult1layer <- expand.grid(layer1=c(16,32,64,128),
                                    kernel=3,
                                    layerdense=c(16,32,64,128),
                                    epochs=20,
                                    exclude=NA,
                                    runtime_mins=NA,
                                    loss=NA,
                                    accuracy=NA)
tuneGridResult1layer$exclude <- 
  tuneGridResult1layer$layerdense<tuneGridResult1layer$layer1
tuneGridResult1layer <- 
  tuneGridResult1layer %>% 
  filter(exclude==FALSE) %>%
  select(-exclude)

# append model performance columns to the tuneGrid 
for (i in 1:nrow(tuneGridResult1layer)){
  tuneGridResult1layer[i,5:7] <- get_result_1layer(i, tuneGridResult1layer[i,1:4])
}

############################################################################################################
########################################----- best tuned -----##############################################

bestTunelayer1 <- tuneGridResult2layer$layer1[which.max(tuneGridResult2layer$accuracy)]
bestTunelayer2 <- tuneGridResult2layer$layer2[which.max(tuneGridResult2layer$accuracy)]
bestTunelayerdense <- tuneGridResult2layer$layerdense[which.max(tuneGridResult2layer$accuracy)]

# the best tuned 2-layer CNN
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = bestTunelayer1, 
                kernel_size = c(3,3), 
                activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = bestTunelayer2, 
                kernel_size = c(3,3), 
                activation = "relu") %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = bestTunelayerdense, 
              kernel_regularizer = regularizer_l2(0.001), 
              activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")
model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))

# train using fashion_mnist
history_cnn_best_tuned <- model %>% fit(mnist_f_train_images, 
                                        mnist_f_train_labels, 
                                        validation_data = list(mnist_f_test_images, 
                                                               mnist_f_test_labels), 
                                        epochs = 50, batch_size = 64)

# plot the loss and accuracy against epochs
plot(history_cnn_best_tuned) + ggtitle("Best tuned CNN on fashion_mnist")

# save the model
model_CNN_best_tuned <- model
model_CNN_best_tuned %>% save_model_hdf5("model_CNN_best_tuned.h5")
load_model_hdf5("model_CNN_best_tuned.h5")

# train using tradition mnist
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = bestTunelayer1, 
                kernel_size = c(3,3), 
                activation = "relu", input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = bestTunelayer2, 
                kernel_size = c(3,3), 
                activation = "relu") %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = bestTunelayerdense, 
              kernel_regularizer = regularizer_l2(0.001), 
              activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")
model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))
history_cnn_best_tuned_mnist <- model %>% fit(mnist_train_images, 
                                              mnist_train_labels, 
                                              validation_data = list(mnist_test_images, 
                                                                     mnist_test_labels), 
                                              epochs = 50, batch_size = 64)

# plot the loss and accuracy against epochs
plot(history_cnn_best_tuned_mnist) + ggtitle("Best tuned CNN on tradition mnist")

# compare models on datasets
compare <- data.frame(mnist=c(0,0,0),fashion_mnist=c(0,0,0),delta_dataset=c(0,0,NA))
row.names(compare) <- c("ANN","Best-tuned CNN","delta_model")
compare["ANN","mnist"] <- tail(history_mnist_NN$metrics$val_accuracy,1)
compare["ANN","fashion_mnist"] <- tail(history_mnist_f_NN$metrics$val_accuracy,1)
compare["Best-tuned CNN","mnist"] <- tail(history_cnn_best_tuned_mnist$metrics$val_accuracy,1)
compare["Best-tuned CNN","fashion_mnist"] <- tail(history_cnn_best_tuned$metrics$val_accuracy,1)
compare$delta_dataset <- round(compare$fashion_mnist-compare$mnist,4)
compare["delta_model",] <- round(compare["Best-tuned CNN",]-compare["ANN",],4)
compare["delta_model","delta_dataset"] <- NA

# save the workspace for RMarkDown knitting
save.image("workspace.RData")
