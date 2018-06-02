#########################
#Data preparation
#########################

library(dplyr)
library(caret)


Credit.train <- read.csv("AT3_credit_train_STUDENT.csv")

head(Credit.train)
str(Credit.train)#23101 obs. of  17 variables
summary(Credit.train) #One of the observations has maximum age of 141. There is a chance for data impurity.
sapply(Credit.train[,c(3,4,5,6,17)],function(x)unique(x)) #Find values in each column of a data frame.

Credit.train %>% filter(SEX == "dolphin" | SEX== "cat" | SEX=="dog") #Check the number of rows containing undefined levels in SEX.
Credit.train <- Credit.train[Credit.train$SEX==1 | Credit.train$SEX==2, ] #Retain Sex=1 and Sex=2.
Credit.train$SEX <- as.integer(Credit.train$SEX)
# Credit.train <- Credit.train[Credit.train$MARRIAGE != 0,] #Remove rows with undefined level 0.
# 
# Credit.train %>% filter(EDUCATION == 0) #There are 11 rows with education 0.
# Credit.train <- Credit.train[Credit.train$EDUCATION != 0,] #Remove education with undefined level 0.

Credit.train$EDUCATION[Credit.train$EDUCATION==6] <- 5 # Levels 5 and 6 of education are unknown. Merge them as a single level.

as.character(levels(Credit.train$default)[1]) #Check the level of first value.
as.character(Credit.train$default[1])#The level and values for default do not match.
Credit.train$default <- as.character(Credit.train$default)
Credit.train$default[Credit.train$default=='N'] <- 0 #Assign 1 for N
Credit.train$default[Credit.train$default=='Y'] <- 1 #Assign 0 for Y
sapply(Credit.train[,c(3,4,5,6,17)],function(x)unique(x)) #Check the columns are in appropriate format.

sum(sapply(Credit.train[,c(3,4,5,6,17)],function(x)is.na(x))) #Check it any columns contain NA.

Credit.train$LIMIT_BAL <- as.integer(Credit.train$LIMIT_BAL) #Convert LIMIT_BAL to integer.
Credit.train$default <- as.integer(Credit.train$default)

#The maximum age of observations in validation set is 79. Filter observations till age 79.
Credit.train %>% filter(AGE > 79) #There are 50 observations with abnormal ages.
Credit.train <- Credit.train[Credit.train$AGE <=79,]
dim(Credit.train) #22999 observations and 17 variables.


# Credit.cor <- cor(Credit.train)
# corrplot::corrplot(Credit.cor)
# class(Credit.train$EDUCATION)

#Validation set
Credit.test <- read.csv("AT3_credit_test_STUDENT.csv")
summary(Credit.test)
str(Credit.test)
# sapply(Credit.test[,c(3,4,5,6)],function(x)unique(x)) #There is level 0 in Education and Marriage.
# Credit.test %>% filter(EDUCATION == 0) #There are just 3 rows with level 0 in validation set.
# Credit.test <- Credit.test[Credit.test$EDUCATION != 0,] #Remove education with undefined level 0.
Credit.test$EDUCATION[Credit.test$EDUCATION==6] <- 5
# Credit.test %>% filter(MARRIAGE == 0) #16 rows with marriage =0. 
# Credit.test <- Credit.test[Credit.test$MARRIAGE != 0,]
sapply(Credit.test[,c(3,4,5,6)],function(x)unique(x))
max(Credit.test$AGE)


# Train-test split
set.seed(1)
train.index <- sample(1:nrow(Credit.train),round(nrow(Credit.train)*0.7),replace = F)
trainset <- Credit.train[train.index,-which(names(Credit.train) %in% c("ID"))]
testset <- Credit.train[-train.index,-which(names(Credit.train) %in% c("ID"))]
