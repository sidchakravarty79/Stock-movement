library("ggplot2")
library("gridExtra")
library("dplyr")
library("e1071")
library("FBN")
library("caret")
library("fpc")
library("mvoutlier")
library("som")
library("DMwR")
library("dtt")
library("plotly")

################################################
# The code above will load any missing packages required
# Delete any predictions.csv file in the working folder
if (file.exists("predictions.csv") == T) {file.remove("predictions.csv")}

#Save stock_returns_base150.csv file in the working directory
train<-read.csv("stock_returns_base150.csv",header = T,nrows = 50,stringsAsFactors = T)
test<-read.csv("stock_returns_base150.csv",header = T,skip = 50,nrows = 50,stringsAsFactors = T)
colnames(test)<-colnames(train)

# Pair-wise Linear regression
plot(train[,-1])
cor.test<-corr.test(train[,-1])
pairs.panels(train[,-1],show.points =FALSE,scale = T,ellipses = F,lm=T,pch=".", cor=T,
             cex.cor = 0.9)

# LM Model 1 using S2 through S10 as indepdent variables
lm_model1<-lm(S1~.,data=train[,-1])
summary(lm_model1)
modelRMSE1 <- rmse(lm_model1$residuals)   
plot(lm_model2)

# LM Model 2 using S2 and S8 as indepdent variables
lm_model2<-lm(S1~S2+S8,data=train[,-1])
summary(lm_model2)
modelRMSE2 <- rmse(lm_model2$residuals)   

# LM Model 3 using S2,S3,S7 and S8 as indepdent variables
lm_model3<-lm(S1~S2+S3+S7+S8,data=train[,-1])
summary(lm_model3)
modelRMSE3 <- rmse(lm_model3$residuals)   

# LM Model 4 using S2,S3,S6,S7 and S8 as indepdent variables
lm_model4<-lm(S1~S2+S3+S6+S7+S8,data=train[,-1])
summary(lm_model4)
modelRMSE4 <- rmse(lm_model4$residuals)   

# ANOVA comparison
compare1_2<-anova(lm_model1,lm_model2)
compare2_3<-anova(lm_model2,lm_model3)
compare3_4<-anova(lm_model3,lm_model4)

# Predict S1 based on train data
predict_lm_train<-predict(lm_model3,train[,-1],se.fit = T,residuals=T)
R2(predict_lm$fit,train$S1)
RMSE(predict_lm$fit,train$S1)

# Plot model vs acual S1 values
dates <- as.Date(train$date,'%m/%d/%Y')
ggplot(train,aes(dates,train$S1)) + geom_path(colour="red") +
  geom_path(aes(dates,predict_lm_train$fit),colour="blue") +
  scale_y_continuous(name = "S1") + scale_x_date(name = "Date",date_breaks = "1.5 weeks")

# Predict S1 from test data using LM model
predict_lm_test<-predict(lm_model3,test,se.fit = T,residuals=T,na.action = na.omit)

# RBF kernel SVM model tuning grid 1
tune_model<-tune.svm(S1~S2+S3+S7+S8,data = train[,-1],type="eps",kernel="radial",
                     cost= seq(1,1000,by=5),gamma =seq(0.1,20,by=0.5), 
                     tunecontrol= tune.control(sampling ="cross",cross = 10))
tune.model<-as.data.frame(tune_model$performances)

plot_ly(x=tune.model$gamma,y=tune.model$cost,z=tune.model$error,type="contour") %>%
  layout(xaxis=list(title="Gamma"),yaxis=list(title="Cost"),legend=list(title="error"))

train_svm<-svm(S1~S2+S3+S7+S8,data = train[,-1],type= "eps", 
                kernel="radial", cost=tune_model$best.parameters$cost,
                gamma = tune_model$best.parameters$cost,probability=T)

train_p<-predict(train_svm,train[,-1],probability = T)
ggplot(train,aes(dates,train$S1)) + geom_path(col="red") +
  geom_path(aes(dates,train_p),col="blue") +
  scale_y_continuous(name = "S1 predict") + scale_x_date(name = "Date")

RMSE_model<-RMSE(train_p,train$S1)
R2_model<-R2(train_p,train$S1)

# RBF kernel SVM model tuning grid 2
tune_model2<-tune.svm(S1~S2+S3+S7+S8,data = train[,-1],type="eps",kernel="radial",
                     cost= seq(1,10,by=0.5),gamma =seq(0.01,1,by=0.01), 
                     tunecontrol= tune.control(sampling ="cross",cross = 10))
tune.model2<-as.data.frame(tune_model2$performances)

plot_ly(x=tune.model2$gamma,y=tune.model2$cost,z=tune.model2$error,type="contour") %>%
  layout(xaxis=list(title="Gamma"),yaxis=list(title="Cost"),legend=list(title="error"))

train_svm2<-svm(S1~S2+S3+S7+S8,data = train[,-1],type= "eps", kernel="radial", 
                cost=tune_model2$best.parameters$cost,gamma = tune_model2$best.parameters$cost,
                probability=T)
train_p2<-predict(train_svm2,train[,-1],probability = T)

ggplot(train,aes(dates,train$S1)) + geom_path(col="red") +
  geom_path(aes(dates,train_p2),col="blue") +
  scale_y_continuous(name = "S1 predict") + scale_x_date(name = "Date")

RMSE_model2<-RMSE(train_p2,train$S1)
R2_model2<-R2(train_p2,train$S1)

# RBF kernel SVM model tuning grid 3
tune_model3<-tune.svm(S1~S2+S3+S7+S8,data = train[,-1],type="eps",kernel="radial",
                      gamma= seq(0.001,0.5,by=0.001),cost =seq(0.1,7,by=0.5),
                      tunecontrol= tune.control(sampling ="cross",cross = 10))
tune.model3<-as.data.frame(tune_model3$performances)

plot_ly(x=tune.model3$gamma,y=tune.model3$cost,z=tune.model3$error,type="contour") %>%
  layout(xaxis=list(title="Gamma"),yaxis=list(title="Cost"),legend=list(title="error"))

train_svm3<-svm(S1~S2+S3+S7+S8,data = train[,-1],type= "eps", 
               kernel="radial", cost=4.6,gamma = 0.004,probability=T)

train_p3<-predict(train_svm3,train[,-1],probability = T)
ggplot(train,aes(dates,train$S1)) + geom_path(col="red") +
  geom_path(aes(dates,train_p3),col="blue") +
  scale_y_continuous(name = "S1 predict") + scale_x_date(name = "Date")

RMSE_model3<-RMSE(train_p3,train$S1)
R2_model3<-R2(train_p3,train$S1)

# Summary of SVM results
rmse<-rbind(RMSE_model,RMSE_model2,RMSE_model3)
r2<-rbind(R2_model,R2_model2,R2_model3)
summary<-cbind.data.frame(rmse,r2)

# Predict S1 from test data using SVM model 2
predict_svm<-predict(train_svm2,test[,-1])
testdate <- as.Date(test$date,'%m/%d/%Y')
train$S1<-as.data.frame(train$S1)

# Write S1 output to predictions.csv
# Linear model plot
ggplot(test,aes(testdate,predict_lm_test$fit)) + geom_path(colour="red") +
  geom_path(aes(dates,train$S1),colour="blue") +
  scale_y_continuous(name = "S1 predict") + scale_x_date(name = "Date")

# SVM model plot
ggplot(test,aes(testdate,predict_svm)) + geom_path(colour="red") +
  geom_path(aes(dates,train$S1),colour="blue") +
  scale_y_continuous(name = "S1") + scale_x_date(name = "Date",date_breaks= "2 weeks")

svm_output<-cbind(as.data.frame(test$date),predict_svm)
colnames(svm_output)<-c("Date","Value")
write.csv(svm_output,file="predictions.csv",append = F)

#### END######
