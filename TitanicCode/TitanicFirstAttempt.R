#Delete all existing objects in the working environment
rm(list = ls())


#Check working directory
getwd()

#Set working directory to kaggle - titanic folder
setwd("")

#Load relevant libraries
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')
install.packages('randomForest')
library(dplyr)
library(magrittr)
library(tidyverse)
library(ggplot2)
library(Hmisc)
library(psych)
library(readr)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(party)

#Read train and test data set
train <- read.csv("train.csv", stringsAsFactors = TRUE, header = TRUE)
test <- read.csv("test.csv", stringsAsFactors = TRUE, header = TRUE)

#Initial exploration of the data set
str(train)
Hmisc::describe(train)
#891 observations with 12 variables

train <- as_tibble(train) #convert to tibble

summary(train) #Large percentage of passengers were male
#Nulls observed in Age

#Exploratory Data Analysis
#Visualize relationship of survival vs multiple variables
#History tells us that the incident was famous for trying to rescue women and
#children firt. Lets check our data to see if that is actually true

prop.table(table(train$Survived)) #38.38% survived

prop.table(table(train$Sex, train$Survived),1)  #Looks like a larger percentage
#of women did survive (74.2%)

prop.table(table(train$Pclass))

prop.table(table(train$Pclass, train$Survived),1) #Comparitively speaking though
#the first class had a higher percentage of surving they only represent 24% of 
#the total passengers in the training set

#Test the hypothesis for children. Perform wrangling to fill up the NAs in age
pos_com <- regexpr(",", train$Name)
pos_stop <- regexpr("\\.", train$Name)
train <- train %>% mutate(prefix = substr(train$Name, pos_com+2, pos_stop-1))
table(train$prefix)


train %>% group_by(prefix) %>% summarise(mean = mean(Age, na.rm = TRUE))
#Safely assume that Master can be considered as a child. In case of "miss" 
#we may need to explore more

age_chk <- train %>% filter(train$prefix == "Miss", !is.na(train$Age)) 
age_chk %>% mutate(Parch = as.character(Parch)) %>% group_by(Parch) %>% 
summarise(avg_Age = mean(Age,na.rm = TRUE))
#If prefix is miss and parch is not 0 then we can assume the passenger to be a
#child

#Flag 1 for kids 0 for adults

train$flag <- 0
train$flag[train$prefix == "Master" | train$Age < 18 |
        (train$prefix == "Miss" & train$Parch != 0)] <- 1

prop.table(table(train$flag,train$Survived),1) 

#Proving that more children survived....but only marginally

#Additional variable for family size
train$famsize <- 0 
train$famsize <- train$SibSp + train$Parch
prop.table(table(train$famsize,train$Survived),1) 

#Check impact of Fare on survival
ggplot(data = train, aes(x= as.factor(Survived), y = Fare , 
                         color = as.factor(Survived))) + geom_boxplot()

#Wrangle test data
pos_com <- regexpr(",", test$Name)
pos_stop <- regexpr("\\.", test$Name)
test <- test %>% mutate(prefix = substr(test$Name, pos_com+2, pos_stop-1))
test$flag <- 0
test$flag[test$prefix == "Master" | test$Age < 18 |
             (test$prefix == "Miss" & test$Parch != 0)] <- 1
test$famsize <- 0 
test$famsize <- test$SibSp + test$Parch
test$Fare[is.na(test$Fare)] <- median(test$Fare,na.rm = TRUE)
summary(test$Fare)

#Modelling  - Logistic
fit.lg <- glm(Survived ~ Pclass + Sex + famsize + flag + Fare,  
              data = train,family = "binomial")

summary(fit.lg)
lg.probs <- predict(fit.lg,test, type = "response")
lg.pred <- rep(0,418)
lg.pred[lg.probs > 0.5] = 1
lg_submit <- data.frame(PassengerId = test$PassengerId, Survived = lg.pred)
write.csv(lg_submit, file = "logreg.csv", row.names = FALSE)

lg_tab <- table(lg.pred,train$Survived)
pre_power_lg <- sum(diag(lg_tab))/sum(lg_tab)


#Modelling - Decision Tree
fit.dec <- rpart(Survived ~ Pclass + Sex + famsize + flag + Fare, 
                 data = train,method = "class")

plot(fit.dec)
text(fit.dec)

fancyRpartPlot(fit.dec)
dec.pred <- predict(fit.dec, test, type = "class")
dec_submit <- data.frame(PassengerId = test$PassengerId, Survived = dec.pred)
write.csv(dec_submit, file = "dec.csv", row.names = FALSE)

dec_tab <- table(dec.pred,train$Survived)
pre_power_dec <- sum(diag(dec_tab))/sum(dec_tab)

#Modelling - Random Forest
set.seed(415)

fit.rf <- randomForest(as.factor(Survived) ~ Pclass + Sex + famsize + flag + Fare, 
                       data = train, importance = TRUE, ntree = 2000)


varImpPlot(fit.rf)

rf.pred <- predict(fit.rf,test)
rf_submit <- data.frame(PassengerId = test$PassengerId, Survived = rf.pred)
write.csv(rf_submit, file = "rf.csv", row.names = FALSE)



rf_tab <- table(rf.pred, train$Survived)
pre_power_rf <- sum(diag(rf_tab))/sum(rf_tab)

#Pre - processing before ensemble modelling 
train$flag1 <- "train"
test$flag1 <- "test"
test$Survived <- 0
combi <- rbind(train,test)
combi$prefix <- as.factor(combi$prefix)
train1 <- combi[combi$flag1 == "train",]
train$prefix <- as.factor(train$prefix)
str(train1$prefix)

#Modelling - Ensemble modelling
fit.ens <- cforest(as.factor(Survived) ~ Pclass + Sex + famsize + flag + 
                     Fare + prefix+SibSp + Parch+Embarked, 
                        data = train1, controls = cforest_unbiased(ntree = 2000, 
                                                                  mtry =3))

ens.pred <- predict(fit.ens, test1,OOB = TRUE, type = "response")
ens_submit <- data.frame(PassengerId = test$PassengerId, Survived = ens.pred)
write.csv(ens_submit, file = "ens.csv", row.names = FALSE)


test$prefix <- as.factor(test$prefix)
ens_tab <- table(ens.pred, train$Survived)
pre_power_ens <- sum(diag(ens_tab))/sum(ens_tab)












