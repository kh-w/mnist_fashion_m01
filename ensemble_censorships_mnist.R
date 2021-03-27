library(keras)
library(dplyr)
library(ggplot2)
library(caret)

######################################## 2-layer CNN ######################################## 

callmodel <- function(){
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 64, 
                  kernel_size = c(3,3), 
                  activation = "relu", input_shape = c(28,28,1)) %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_conv_2d(filters = 64, 
                  kernel_size = c(3,3), 
                  activation = "relu") %>%
    layer_flatten() %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 128, 
                kernel_regularizer = regularizer_l2(0.001), 
                activation = "relu") %>%
    layer_dense(units = 10, activation = "softmax")
  
  model %>% compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = c("accuracy"))
}

###############################--- import dataset ---########################################
# load the mnist dataset
mnist <- dataset_mnist()
mnist_train_images <- mnist$train$x
mnist_train_labels <- mnist$train$y
mnist_test_images <- mnist$test$x
mnist_test_labels <- mnist$test$y
mnist_train_labels <- to_categorical(mnist_train_labels)
mnist_test_labels <- to_categorical(mnist_test_labels)
mnist_test_labels_raw <- unlist(lapply(split(matrix(mnist_test_labels,10000,10),1:10000),which.max))-1

# reshape and standardize
mnist_train_images <- array_reshape(mnist_train_images, c(60000, 784))
mnist_train_images <- mnist_train_images / 255
mnist_test_images <- array_reshape(mnist_test_images, c(10000, 784))
mnist_test_images <- mnist_test_images / 255

# apply censors
mnist_train_dat_raw <- mnist_train_images
mnist_test_dat_raw <- mnist_test_images

first_censor <- 1:196
mnist_train_images_1 <- mnist_train_images
mnist_train_images_1[,first_censor] <- 1
mnist_test_images_1 <- mnist_test_images
mnist_test_images_1[,first_censor] <- 1

second_censor <- 197: 392
mnist_train_images_2 <- mnist_train_images
mnist_train_images_2[,second_censor] <- 1
mnist_test_images_2 <- mnist_test_images
mnist_test_images_2[,second_censor] <- 1

third_censor <- 393:590
mnist_train_images_3 <- mnist_train_images
mnist_train_images_3[,third_censor] <- 1
mnist_test_images_3 <- mnist_test_images
mnist_test_images_3[,third_censor] <- 1

fourth_censor <- 589:784
mnist_train_images_4 <- mnist_train_images
mnist_train_images_4[,fourth_censor] <- 1
mnist_test_images_4 <- mnist_test_images
mnist_test_images_4[,fourth_censor] <- 1

# reshape and standardize mnist for CNN
mnist_train_images_1 <- array_reshape(mnist_train_images_1, c(60000, 28, 28, 1))
mnist_train_images_2 <- array_reshape(mnist_train_images_2, c(60000, 28, 28, 1))
mnist_train_images_3 <- array_reshape(mnist_train_images_3, c(60000, 28, 28, 1))
mnist_train_images_4 <- array_reshape(mnist_train_images_4, c(60000, 28, 28, 1))
mnist_test_images_1 <- array_reshape(mnist_test_images_1, c(10000, 28, 28, 1))
mnist_test_images_2 <- array_reshape(mnist_test_images_2, c(10000, 28, 28, 1))
mnist_test_images_3 <- array_reshape(mnist_test_images_3, c(10000, 28, 28, 1))
mnist_test_images_4 <- array_reshape(mnist_test_images_4, c(10000, 28, 28, 1))

#############################################################################################

# call and compile model, censor 1
model <- callmodel()
model %>% fit(mnist_train_images_1, 
              mnist_train_labels, 
              validation_data = list(mnist_test_images_1, 
                                     mnist_test_labels), 
              epochs = 20, batch_size = 64)
result1 <- model %>% predict_classes(x=mnist_test_images_1)

# store the result for censor 1
result.df <- data.frame(real_lab=factor(mnist_test_labels_raw),censor1=factor(result1))

#############################################################################################

# call and compile model, censor 2
model <- callmodel()
model %>% fit(mnist_train_images_2, 
              mnist_train_labels, 
              validation_data = list(mnist_test_images_2, 
                                     mnist_test_labels), 
              epochs = 20, batch_size = 64)
result2 <- model %>% predict_classes(x=mnist_test_images_2)

# store the result for censor 2
result.df$censor2 <- factor(result2)

#############################################################################################

# call and compile model, censor 3
model <- callmodel()
model %>% fit(mnist_train_images_3, 
              mnist_train_labels, 
              validation_data = list(mnist_test_images_3, 
                                     mnist_test_labels), 
              epochs = 20, batch_size = 64)
result3 <- model %>% predict_classes(x=mnist_test_images_3)

# store the result for censor 3
result.df$censor3 <- factor(result3)

#############################################################################################

# call and compile model, censor 4
model <- callmodel()
model %>% fit(mnist_train_images_4, 
              mnist_train_labels, 
              validation_data = list(mnist_test_images_4, 
                                     mnist_test_labels), 
              epochs = 20, batch_size = 64)
result4 <- model %>% predict_classes(x=mnist_test_images_4)

# store the result for censor 4
result.df$censor4 <- factor(result4)

#############################################################################################

# ensemble

ensemble <- function(vec){
  x <- sapply(0:9,function(i){sum(vec==i)})
  which.max(x)-1
}

result.df.list <- split(result.df,seq(nrow(result.df)))

result.df$ensemble <- factor(unlist(lapply(result.df.list,ensemble)))

confusionMatrix(result.df$real_lab, result.df$censor1)$overall["Accuracy"]
confusionMatrix(result.df$real_lab, result.df$censor2)$overall["Accuracy"]
confusionMatrix(result.df$real_lab, result.df$censor3)$overall["Accuracy"]
confusionMatrix(result.df$real_lab, result.df$censor4)$overall["Accuracy"]

# ensemble accuracy

confusionMatrix(result.df$real_lab, result.df$ensemble)$overall["Accuracy"]
