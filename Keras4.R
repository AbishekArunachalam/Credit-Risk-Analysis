# install.packages("dplyr")

library(dplyr)
Credit.train <- read.csv("AT3_credit_train_STUDENT.csv")
Credit.test <- read.csv("AT3_credit_test_STUDENT.csv")


#Clean LIMIT_BAL
Credit.train <- Credit.train[Credit.train$LIMIT_BAL >0,]
Credit.train$LIMIT_BAL <- as.integer(Credit.train$LIMIT_BAL) #Convert LIMIT_BAL to integer.

#Clean SEX 
Credit.train %>% filter(SEX == "dolphin" | SEX== "cat" | SEX=="dog")
Credit.train <- Credit.train[Credit.train$SEX==1 | Credit.train$SEX==2, ] #Retain Sex=1 and Sex=2.
Credit.train$SEX <-  as.integer(Credit.train$SEX)
Credit.train$SEX <-  as.factor(Credit.train$SEX)

#Clean EDUCATION
Credit.train$EDUCATION[Credit.train$EDUCATION==6 | Credit.train$EDUCATION==0] <- 5 # Levels 5 and 6 of education are unknown. Merge them as a single level.
Credit.train$EDUCATION <-  as.factor(Credit.train$EDUCATION)

#Clean MARRIAGE
Credit.train$MARRIAGE[Credit.train$MARRIAGE==0] <- 3 #Set unknown level marriage 0 to 3-Other. 
Credit.train$MARRIAGE <-  as.factor(Credit.train$MARRIAGE)

#Clean AGE
Credit.train %>% filter(AGE > 79) #There are 50 observations with abnormal ages. There are no observations for age between 79-128.
Credit.train <- Credit.train[Credit.train$AGE <=79,]

#Clean Default
Credit.train$default <- as.character(Credit.train$default)
Credit.train$default[Credit.train$default=='N'] <- 0 #Assign 0 for N
Credit.train$default[Credit.train$default=='Y'] <- 1 #Assign 1 for Y
Credit.train$default <- as.integer(Credit.train$default)

##Cred.test
Credit.test$EDUCATION[Credit.test$EDUCATION==6 | Credit.test$EDUCATION==0] <- 5
Credit.test$MARRIAGE[Credit.test$MARRIAGE==0] <- 3
Credit.test$LIMIT_BAL <- as.integer(Credit.test$LIMIT_BAL) #Convert LIMIT_BAL to integer.
Credit.test$SEX <- as.factor(Credit.test$SEX)
Credit.test$EDUCATION <- as.factor(Credit.test$EDUCATION)
Credit.test$MARRIAGE <- as.factor(Credit.test$MARRIAGE)

set.seed(1)
train.index <- sample(1:nrow(Credit.train),round(nrow(Credit.train)*0.8),replace = F)
trainset <- Credit.train[train.index,-which(names(Credit.train) %in% c("ID"))]
testset <- Credit.train[-train.index,-which(names(Credit.train) %in% c("ID"))]

library(caret)
#Hot coding variables in trainset
dummytrain <- dummyVars(" ~ .", data = trainset)
trainsetdummy <- data.frame(predict(dummytrain, newdata = trainset))
y_train  <- trainsetdummy[,which(names(trainsetdummy) %in% c("default"))]

#Hot coding variables in testset
dummytest <- dummyVars(" ~ .", data = testset)
testsetdummy <- data.frame(predict(dummytest, newdata = testset))
y_test <- testsetdummy[,which(names(testsetdummy) %in% c("default"))]

#Scaling all predictors 
library(recipes)
rec_obj <- 
  recipe (default ~ ., data = trainsetdummy) %>%
  step_center (all_predictors(), -all_outcomes()) %>%
  step_scale (all_predictors(), -all_outcomes()) %>%
  prep (data = trainsetdummy)

x_train_tbl <- 
  bake (rec_obj, newdata = trainsetdummy) %>% 
  select (-default)
x_train_tbl <- as.matrix(x_train_tbl)

x_test_tbl <- 
  bake (rec_obj, newdata = testsetdummy) %>% 
  select (-default)
x_test_tbl <- as.matrix(x_test_tbl)

library(keras)


# Building our Artificial Neural Network
model <- keras_model_sequential()

model %>%
  # First hidden layer
  layer_dense(
    units              = 32,
    kernel_initializer = "uniform",
    activation         = "relu",
    input_shape        = ncol(x_train_tbl)) %>%
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Second hidden layer
  layer_dense(
    units              = 32,
    kernel_initializer = "uniform",
    activation         = "relu") %>%
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Output layer
  layer_dense(
    units              = 1,
    kernel_initializer = "uniform",
    activation         = "sigmoid") %>%
  # Compile ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )


# Fit the keras model to the training data
fit_keras <- fit(
  object           = model,
  x                = x_train_tbl,
  y                = y_train,
  batch_size       = 50,
  epochs           = 100,################
  validation_split = 0.30
)

#Using the model to predict the testset

pred <- model %>% 
           predict_classes(x_test_tbl)


table(pred)
u = union(pred,y_test)
t = table(factor(pred, u), factor(y_test, u))
KerasConfusinMatrix=confusionMatrix(t)
KerasConfusinMatrix


library(pROC)
auc(as.numeric(pred),as.numeric(y_test))

# install.packages("MLmetrics")
library(MLmetrics)
F1_Score(y_pred = pred, y_true = y_test, positive = "1")





#################################  NEURAL NETWORKS USING NURALNET #########################################
nntrain<-data.frame(x_train_tbl,y_train)

library(neuralnet)
set.seed(1)

neural.model<-neuralnet(y_train~LIMIT_BAL+SEX.1+SEX.2+EDUCATION.1+EDUCATION.2+EDUCATION.3+EDUCATION.4+EDUCATION.5+
                          MARRIAGE.1+MARRIAGE.2  +MARRIAGE.3+AGE+PAY_PC1+PAY_PC2+PAY_PC3+AMT_PC1+AMT_PC2+AMT_PC3+
                          AMT_PC4+AMT_PC5+AMT_PC6+AMT_PC7,data=nntrain,
                        hidden=5,threshold=0.01,err.fct="sse",linear.output=FALSE,likelihood=TRUE,
                        stepmax=1e+05,rep=1,startweights=NULL,learningrate.limit=list(0.1,1.5),
                        learningrate.factor=list(minus=0.5,plus=1.5),learningrate=0.5,lifesign="minimal",
                        lifesign.step=1000,algorithm="backprop",act.fct="logistic",exclude=NULL,constant.weights=NULL)

summary(neural.model)
plot(neural.model)
#Predict Close Price
testnew.nnbp<-compute(neural.model,x_test_tbl)


confusionMatrix(data = testnew.nnbp, reference = y_test)


table(testnew.nnbp)
u = union(testnew.nnbp,y_test)
t = table(factor(testnew.nnbp, u), factor(y_test, u))
NNConfusinMatrix=confusionMatrix(t)
NNConfusinMatrix



