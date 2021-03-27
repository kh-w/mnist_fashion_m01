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

###############################--- define censors ---########################################

first_censor <- 2:197
second_censor <- 198:393
third_censor <- 394:589
fourth_censor <- 590:785

########################--- import fashion mnist dataset ---#################################

# load the fashion_mnist dataset
mnist_f_train_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_train.csv')
mnist_f_test_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_test.csv')

# apply censors
mnist_f_train_dat1 <- mnist_f_train_dat
mnist_f_test_dat1 <- mnist_f_test_dat
mnist_f_train_dat1[,first_censor] <- 255
mnist_f_test_dat1[,first_censor] <- 255

mnist_f_train_dat2 <- mnist_f_train_dat
mnist_f_test_dat2 <- mnist_f_test_dat
mnist_f_train_dat2[,second_censor] <- 255
mnist_f_test_dat2[,second_censor] <- 255

mnist_f_train_dat3 <- mnist_f_train_dat
mnist_f_test_dat3 <- mnist_f_test_dat
mnist_f_train_dat3[,third_censor] <- 255
mnist_f_test_dat3[,third_censor] <- 255

mnist_f_train_dat4 <- mnist_f_train_dat
mnist_f_test_dat4 <- mnist_f_test_dat
mnist_f_train_dat4[,fourth_censor] <- 255
mnist_f_test_dat4[,fourth_censor] <- 255

#############################################################################################

# reshape the datasets
# 1
mnist_f_train_images_1 <- array(as.numeric(as.matrix(mnist_f_train_dat1[,2:785],nrow=60000,ncol=784)),dim=c(60000,28,28,1))
mnist_f_test_images_1 <- array(as.numeric(as.matrix(mnist_f_test_dat1[,2:785],nrow=10000,ncol=784)),dim=c(10000,28,28,1))
# 2
mnist_f_train_images_2 <- array(as.numeric(as.matrix(mnist_f_train_dat2[,2:785],nrow=60000,ncol=784)),dim=c(60000,28,28,1))
mnist_f_test_images_2 <- array(as.numeric(as.matrix(mnist_f_test_dat2[,2:785],nrow=10000,ncol=784)),dim=c(10000,28,28,1))
# 3
mnist_f_train_images_3 <- array(as.numeric(as.matrix(mnist_f_train_dat3[,2:785],nrow=60000,ncol=784)),dim=c(60000,28,28,1))
mnist_f_test_images_3 <- array(as.numeric(as.matrix(mnist_f_test_dat3[,2:785],nrow=10000,ncol=784)),dim=c(10000,28,28,1))
# 4
mnist_f_train_images_4 <- array(as.numeric(as.matrix(mnist_f_train_dat4[,2:785],nrow=60000,ncol=784)),dim=c(60000,28,28,1))
mnist_f_test_images_4 <- array(as.numeric(as.matrix(mnist_f_test_dat4[,2:785],nrow=10000,ncol=784)),dim=c(10000,28,28,1))
# labels
mnist_f_train_labels <- array(as.integer(as.matrix(mnist_f_train_dat_raw[,1],nrow=60000,ncol=1)))
mnist_f_test_labels <- array(as.integer(as.matrix(mnist_f_test_dat_raw[,1],nrow=10000,ncol=1)))

# format the labels such that it is readable by keras
mnist_f_train_labels_raw <- mnist_f_train_labels
mnist_f_test_labels_raw <- mnist_f_test_labels
mnist_f_train_labels <- to_categorical(mnist_f_train_labels)
mnist_f_test_labels <- to_categorical(mnist_f_test_labels)

#############################################################################################

# call and compile model, censor 1
model <- callmodel()
model %>% fit(mnist_f_train_images_1, 
              mnist_f_train_labels, 
              validation_data = list(mnist_f_test_images_1, 
                                     mnist_f_test_labels),
              epochs = 20, batch_size = 64)
result1 <- model %>% predict_classes(x=mnist_f_test_images_1)

# store the result for censor 1
result.df.fashion <- data.frame(real_lab=factor(mnist_f_test_labels_raw),censor1=factor(result1))

#############################################################################################

# call and compile model, censor 2
model <- callmodel()
model %>% fit(mnist_f_train_images_2, 
              mnist_f_train_labels, 
              validation_data = list(mnist_f_test_images_2, 
                                     mnist_f_test_labels), 
              epochs = 20, batch_size = 64)
result2 <- model %>% predict_classes(x=mnist_f_test_images_2)

# store the result for censor 2
result.df.fashion$censor2 <- factor(result2)

#############################################################################################

# call and compile model, censor 3
model <- callmodel()
model %>% fit(mnist_f_train_images_3, 
              mnist_f_train_labels, 
              validation_data = list(mnist_f_test_images_3, 
                                     mnist_f_test_labels), 
              epochs = 20, batch_size = 64)
result3 <- model %>% predict_classes(x=mnist_f_test_images_3)

# store the result for censor 3
result.df.fashion$censor3 <- factor(result3)

#############################################################################################

# call and compile model, censor 4
model <- callmodel()
model %>% fit(mnist_f_train_images_4, 
              mnist_f_train_labels, 
              validation_data = list(mnist_f_test_images_4, 
                                     mnist_f_test_labels), 
              epochs = 20, batch_size = 64)
result4 <- model %>% predict_classes(x=mnist_f_test_images_4)

# store the result for censor 4
result.df.fashion$censor4 <- factor(result4)

#############################################################################################

# ensemble
ensemble <- function(vec){
  x <- sapply(0:9,function(i){sum(vec==i)})
  which.max(x)-1
}
result.df.fashion.list <- split(result.df.fashion,seq(nrow(result.df.fashion)))
result.df.fashion$ensemble <- factor(unlist(lapply(result.df.fashion.list,ensemble)))

# compare the accuracies
confusionMatrix(result.df.fashion$real_lab, result.df.fashion$censor1)$overall["Accuracy"]
confusionMatrix(result.df.fashion$real_lab, result.df.fashion$censor2)$overall["Accuracy"]
confusionMatrix(result.df.fashion$real_lab, result.df.fashion$censor3)$overall["Accuracy"]
confusionMatrix(result.df.fashion$real_lab, result.df.fashion$censor4)$overall["Accuracy"]

# ensemble accuracy
ensemble_acc_fashion_mnist <- confusionMatrix(result.df.fashion$real_lab, result.df.fashion$ensemble)$overall["Accuracy"]
ensemble_acc_fashion_mnist

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

###############################--- import mnist dataset ---##################################

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
mnist_train_images_1 <- mnist_train_images
mnist_train_images_1[,first_censor-1] <- 1
mnist_test_images_1 <- mnist_test_images
mnist_test_images_1[,first_censor-1] <- 1

mnist_train_images_2 <- mnist_train_images
mnist_train_images_2[,second_censor-1] <- 1
mnist_test_images_2 <- mnist_test_images
mnist_test_images_2[,second_censor-1] <- 1

mnist_train_images_3 <- mnist_train_images
mnist_train_images_3[,third_censor-1] <- 1
mnist_test_images_3 <- mnist_test_images
mnist_test_images_3[,third_censor-1] <- 1

mnist_train_images_4 <- mnist_train_images
mnist_train_images_4[,fourth_censor-1] <- 1
mnist_test_images_4 <- mnist_test_images
mnist_test_images_4[,fourth_censor-1] <- 1

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

# compare the accuracies
confusionMatrix(result.df$real_lab, result.df$censor1)$overall["Accuracy"]
confusionMatrix(result.df$real_lab, result.df$censor2)$overall["Accuracy"]
confusionMatrix(result.df$real_lab, result.df$censor3)$overall["Accuracy"]
confusionMatrix(result.df$real_lab, result.df$censor4)$overall["Accuracy"]

# ensemble accuracy
confusionMatrix(result.df$real_lab, result.df$ensemble)$overall["Accuracy"]

# clean up
rm(result1,result2,result3,result4,model,result.df.fashion.list,result.df.list)
