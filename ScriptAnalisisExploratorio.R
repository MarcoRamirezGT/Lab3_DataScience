# Analisis exploratorio

db<-read.csv('train.csv')


x<-c(1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5)
table(x)
summary(x)

summary(as.factor(x))
