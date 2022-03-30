install.packages("plyr")
install.packages("dplyr")
install.packages("caret")
install.packages("repr")
install.packages("glmnet")
library(glmnet)
library(plyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(repr)

library(MASS)
data(Boston)
## 1. Perform linear regression using data set Boston,F-statistic, residual standard error, and adjusted R2.

attach(Boston) 
colnames(Boston) 

my.lm = lm(medv ~ crim + zn + indus + chas + nox + rm + age + dis + rad+ tax + ptratio + black + lstat )
summary(my.lm)


# Perform best subset selection, using adjusted R2 as the metric for selection
library(leaps)
fit.best = regsubsets(medv ~.,Boston,nvmax=19)  
sum.best=summary(fit.best)
names(sum.best) 	
sum.best$adjr2	
which.max(sum.best$adjr2) 
coef(fit.best,11)

# (b) Then, perform forward and backward stepwise selections, 

fit.fwd=regsubsets(medv ~.,Boston,nvmax=13,method="forward") 
sum.fwd=summary(fit.fwd)

sum.fwd$adjr2
which.max(sum.fwd$adjr2) 

coef(fit.fwd,11)

fit.bwd=regsubsets(medv ~.,Boston,nvmax=13,method="backward") 
sum.bwd=summary(fit.bwd)

sum.bwd$adjr2
which.max(sum.bwd$adjr2)

coef(fit.bwd,11)



#3. (a) Perform ridge regression. 

#Get the package and prepare data 

x0=model.matrix(medv ~.,Boston)	
x =x0[,-1]
y =medv

set.seed(1)
train=sample(1:nrow(x), nrow(x)/2)
x.test=x[-train,] 
y.test=y[-train]
grid=10^seq(10,-2,length=100)

ridge.mod=glmnet(x[train,], y[train], alpha=0, lambda=grid) 
summary(ridge.mod)

cv_ridge <- cv.glmnet(x[train,], y[train], alpha = 0, lambda = grid)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda


# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
}

# Prediction and evaluation on train data
predictions_train <- predict(ridge.mod, s = optimal_lambda, newx = x[train,])
eval_results(y[train], predictions_train, x[train,])

# Prediction and evaluation on test data
predictions_test <- predict(ridge.mod, s = optimal_lambda, newx = x.test)
eval_results(y.test, predictions_test, x.test)

#3. (b) Perform LASSO regression in a similar way as that in part (a).

lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=grid) 
cv_lasso <- cv.glmnet(x[train,], y[train], alpha = 1, lambda = grid)
optimal_lambda <- cv_lasso$lambda.min
optimal_lambda

# Prediction and evaluation on train data
predictions_train <- predict(lasso.mod, s = optimal_lambda, newx = x[train,])
eval_results(y[train], predictions_train, x[train,])

# Prediction and evaluation on test data
predictions_test <- predict(lasso.mod, s = optimal_lambda, newx = x.test)
eval_results(y.test, predictions_test, x.test)
