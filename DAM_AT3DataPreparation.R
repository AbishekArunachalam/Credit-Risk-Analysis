library(dplyr)
Credit.train <- read.csv("AT3_credit_train_STUDENT.csv")

head(Credit.train)
str(Credit.train)#23101 obs. of  17 variables
summary(Credit.train) #One of the observations has maximum age of 141. There is a chance for data impurity.
sapply(Credit.train[,c(3,4,5,6,17)],function(x)unique(x)) #Find values in each column of a data frame.

#Clean LIMIT_BAL
#There is min value of -99 in LIMIT_BAL which is not possible.
Credit.train %>% filter(LIMIT_BAL < 0) #There are 50 rows. Lets remove them.
Credit.train <- Credit.train[Credit.train$LIMIT_BAL >0,]
Credit.train$LIMIT_BAL <- as.integer(Credit.train$LIMIT_BAL) #Convert LIMIT_BAL to integer.
summary(Credit.train)

#Clean SEX 
Credit.train %>% filter(SEX == "dolphin" | SEX== "cat" | SEX=="dog") #Check the number of rows containing undefined levels in SEX.
Credit.train <- Credit.train[Credit.train$SEX==1 | Credit.train$SEX==2, ] #Retain Sex=1 and Sex=2.
Credit.train$SEX <- factor(Credit.train$SEX)

#Clean EDUCATION
Credit.train$EDUCATION[Credit.train$EDUCATION==6 | Credit.train$EDUCATION==0] <- 5 # Levels 5 and 6 of education are unknown. Merge them as a single level.
Credit.train$EDUCATION <- as.factor(Credit.train$EDUCATION)

#Clean MARRIAGE
Credit.train$MARRIAGE[Credit.train$MARRIAGE==0] <- 3 #Set unknown level marriage 0 to 3-Other. 
Credit.train$MARRIAGE <- as.factor(Credit.train$MARRIAGE)

#Clean AGE
#The maximum age of observations in validation set is 79. Filter observations till age 79.
Credit.train %>% filter(AGE > 79) #There are 50 observations with abnormal ages. There are no observations for age between 79-128.
Credit.train <- Credit.train[Credit.train$AGE <=79,]

#Clean Default
Credit.train$default <- as.character(Credit.train$default)
Credit.train$default[Credit.train$default=='N'] <- 0 #Assign 0 for N
Credit.train$default[Credit.train$default=='Y'] <- 1 #Assign 1 for Y
Credit.train$default <- as.factor(Credit.train$default)

sapply(Credit.train[,c(3,4,5,6,17)],function(x)unique(x)) #Check the columns are in appropriate format.

sum(sapply(Credit.train[,c(3,4,5,6,17)],function(x)is.na(x))) #Check it any columns contain NA.

dim(Credit.train) #23000 observations and 17 variables.

write.csv(Credit.train,"Clean_Credit_Train.csv",row.names = F)

rm(list=ls())