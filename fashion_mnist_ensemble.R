library(matrixStats)
library(ggplot2)
library(tidyr)
library(caret)
library(dplyr)
library(doParallel)
library(parallel)
library(e1071)
library(kknn)
library(klaR)

detectCores()

cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# Labels
# Each training and test example is assigned to one of the following labels:
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot

labelnames <- c("0 T-shirt/top",
                "1 Trouser",
                "2 Pullover",
                "3 Dress",
                "4 Coat",
                "5 Sandal",
                "6 Shirt",
                "7 Sneaker",
                "8 Bag",
                "9 Ankle boot") 

# setwd("C:/Users/woodr/Google Drive/EdX - Data Science Capstone Project/My own project")
getwd()
train_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_train.csv')

set.seed(2, sample.kind = "Rounding")

# train_dat <- train_dat[sample(1:60000, 10000, replace = FALSE),] # get a subset to do speed run first

test_dat <- read.csv(file = 'datasets/fashion_mnist/fashion-mnist_test.csv')
test_dat <- test_dat %>% mutate(label = factor(label))

nearZV <- nearZeroVar(train_dat[,2:785])

train_dat <- train_dat[,-(nearZV+1)] %>% mutate(label = factor(label))

## Random forest

tic <- Sys.time()

ctrl <- trainControl(method="cv", number = 5)
set.seed(3, sample.kind = "Rounding")
mtry <- seq(1,floor(sqrt(ncol(train_dat))),2)
tunegrid <- expand.grid(.mtry=mtry)

rf.train <- train(label ~ .,
                  data = train_dat,
                  method = "rf",
                  metric = "Accuracy",
                  tuneGrid = tunegrid,
                  trControl = ctrl)

summary(rf.train)

rf_train_pred <- predict.train(rf.train, newdata = train_dat)
confusionMatrix(rf_train_pred, train_dat[,1])$overall["Accuracy"]

rf_test_pred <- predict.train(rf.train, newdata = test_dat)
confusionMatrix(rf_test_pred, test_dat[,1])$overall["Accuracy"]

toc <- Sys.time()
toc - tic

## kNN

tic <- Sys.time()

preProc <- c("center","scale")
kknnBestTune <- data.frame(kmax = 11, distance = 1, kernel = 'optimal')
control_cv <- trainControl(method = "cv", number = 5)

kknn.train <- train(label ~ .,
                    data = train_dat,
                    method = "kknn",
                    preProcess = preProc,
                    tuneGrid = kknnBestTune,
                    trControl = control_cv,
                    metric = "Accuracy")

summary(kknn.train) 

kknn_train_pred <- predict.train(kknn.train, newdata = train_dat)
confusionMatrix(kknn_train_pred, train_dat[,1])$overall["Accuracy"]

kknn_test_pred <- predict.train(kknn.train, newdata = test_dat)
confusionMatrix(kknn_test_pred, test_dat[,1])$overall["Accuracy"]

toc <- Sys.time()
toc - tic

## LDA

tic <- Sys.time()

lda.train <- train(label ~ .,
                   data = train_dat,
                   method = "lda",
                   metric = "Accuracy",
                   preProcess = preProc,
                   trControl = control_cv)

summary(lda.train)

lda_train_pred <- predict.train(lda.train, newdata = train_dat)
confusionMatrix(lda_train_pred, train_dat[,1])$overall["Accuracy"]

lda_test_pred <- predict.train(lda.train, newdata = test_dat)
confusionMatrix(lda_test_pred, test_dat[,1])$overall["Accuracy"]

toc <- Sys.time()
toc - tic

## Naive bayes

tic <- Sys.time()

nb.train <- train(label ~ .,
                  data = train_dat,
                  method = "nb",
                  metric = "Accuracy",
                  preProcess = preProc,
                  trControl = control_cv)

summary(nb.train)

nb_train_pred <- predict.train(nb.train, newdata = train_dat)
confusionMatrix(nb_train_pred, train_dat[,1])$overall["Accuracy"]

nb_test_pred <- predict.train(nb.train, newdata = test_dat)
confusionMatrix(nb_test_pred, test_dat[,1])$overall["Accuracy"]

toc <- Sys.time()
toc - tic

## SVM Linear Kernel

tic <- Sys.time()

svm.train <- train(label ~ .,
                   data = train_dat,
                   method = "svmLinear2",
                   metric = "Accuracy",
                   tuneGrid = data.frame(cost = c(.25, .5, 1)),
                   preProcess = preProc,
                   trControl = control_cv)

summary(svm.train)

svm_train_pred <- predict.train(svm.train, newdata = train_dat)
confusionMatrix(svm_train_pred, train_dat[,1])$overall["Accuracy"]

svm_test_pred <- predict.train(svm.train, newdata = test_dat)
confusionMatrix(svm_test_pred, test_dat[,1])$overall["Accuracy"]

toc <- Sys.time()
toc - tic

## Ensemble

ensemble_train <- data.frame(label = train_dat$label, 
                       lda = lda_train_pred, 
                       naivebayes = nb_train_pred,
                       svm = svm_train_pred,
                       kknn = kknn_train_pred,
                       rf = rf_train_pred)

temp <- data.frame(lda = as.numeric(as.character(unlist(ensemble_train$lda))),
                   naivebayes = as.numeric(as.character(unlist(ensemble_train$naivebayes))),
                   svm = as.numeric(as.character(unlist(ensemble_train$svm))),
                   kknn = as.numeric(as.character(unlist(ensemble_train$kknn))),
                   rf = as.numeric(as.character(unlist(ensemble_train$rf))))

for (i in 1:60000){
  temp$ensemble[i] <- as.numeric(as.character((data.frame(table(as.numeric(temp[i,]))) %>% arrange(desc(Freq)))[1,1]))
}

ensemble_train$ensemble <- factor(temp$ensemble)
rm(temp)

confusionMatrix(ensemble_train$label, ensemble_train$lda)$overall["Accuracy"]
confusionMatrix(ensemble_train$label, ensemble_train$naivebayes)$overall["Accuracy"]
confusionMatrix(ensemble_train$label, ensemble_train$svm)$overall["Accuracy"]
confusionMatrix(ensemble_train$label, ensemble_train$kknn)$overall["Accuracy"]
confusionMatrix(ensemble_train$label, ensemble_train$rf)$overall["Accuracy"]
confusionMatrix(ensemble_train$label, ensemble_train$ensemble)$overall["Accuracy"]

ensemble_test <- data.frame(label = test_dat$label, 
                             lda = lda_test_pred, 
                             naivebayes = nb_test_pred,
                             svm = svm_test_pred,
                             kknn = kknn_test_pred,
                             rf = rf_test_pred)

temp <- data.frame(lda = as.numeric(as.character(unlist(ensemble_test$lda))),
                   naivebayes = as.numeric(as.character(unlist(ensemble_test$naivebayes))),
                   svm = as.numeric(as.character(unlist(ensemble_test$svm))),
                   kknn = as.numeric(as.character(unlist(ensemble_test$kknn))),
                   rf = as.numeric(as.character(unlist(ensemble_test$rf))))

for (i in 1:10000){
  temp$ensemble[i] <- as.numeric(as.character((data.frame(table(as.numeric(temp[i,]))) %>% arrange(desc(Freq)))[1,1]))
}

ensemble_test$ensemble <- factor(temp$ensemble)
rm(temp)

confusionMatrix(ensemble_test$label, ensemble_test$lda)$overall["Accuracy"]
confusionMatrix(ensemble_test$label, ensemble_test$naivebayes)$overall["Accuracy"]
confusionMatrix(ensemble_test$label, ensemble_test$svm)$overall["Accuracy"]
confusionMatrix(ensemble_test$label, ensemble_test$kknn)$overall["Accuracy"]
confusionMatrix(ensemble_test$label, ensemble_test$rf)$overall["Accuracy"]
confusionMatrix(ensemble_test$label, ensemble_test$ensemble)$overall["Accuracy"]

stopCluster(cl)

save.image("ensemble.RData")
