library(keras)
library(mlbench)
library(dplyr)
library(magrittr)
library(neuralnet)

data("BostonHousing")
data <- BostonHousing
str(data)
data %<>% mutate_if(is.factor, as.numeric)
n <- neuralnet(medv ~ crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat,
               data = data,
               hidden = c(12,7),
               linear.output = F,
               lifesign = 'full',
               rep=1)
plot(n,col.hidden = 'darkgreen',     
     col.hidden.synapse = 'darkgreen',
     show.weights = F,
     information = F,
     fill = 'lightblue')
data <- as.matrix(data)
dimnames(data) <- NULL
set.seed(123)
ind <- sample(2, nrow(data), replace = T, prob = c(.7, .3))
training <- data[ind==1,1:13]
test <- data[ind==2, 1:13]
trainingtarget <- data[ind==1, 14]
testtarget <- data[ind==2, 14]
str(trainingtarget)

str(testtarget)


