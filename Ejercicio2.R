#Librerias
library("readxl")
library(tibbletime)
library(dplyr)
library(tidyverse)
library(forecast)
library(tseries)
library(fUnitRoots)
library(ggfortify)
library(lmtest)
library(prophet)
library(zoo)

#Leer xls files
consumo <- read_excel("Consumo.xlsx")

consumo$Fecha<-as.Date(consumo$Fecha, "%Y/%m/%d")
str(consumo)

#Series de tiempo
fecha<-consumo[,'Fecha']
diesel<-consumo[,'Diesel alto azufre']
dieselc<-consumo[c('Fecha','Diesel alto azufre')]

#Diesel
diesel_ts<-ts(dieselc$`Diesel alto azufre`, start = c(2001,1),frequency = 12)
start(diesel_ts)
end(diesel_ts)
frequency(diesel_ts)
plot(diesel_ts)
abline(reg=lm(diesel_ts~time(diesel_ts)), col=c("red"))
plot(aggregate(diesel_ts,FUN=mean))
dec.Diesel<-decompose(diesel_ts)
plot(dec.Diesel)

train<-head(diesel_ts, round(length(diesel_ts) * 0.7))
h<-length(diesel_ts) - length(train)
test<-tail(diesel_ts, h)

#Aplicaremos una transformación logarítmica
logDiesel<-log(train)
plot(decompose(train))
plot(train)

adfTest(train)
unitrootTest(train)

adfTest(diff(train))
unitrootTest(diff(train))

#Gráfico de autocorrelación
acf(logDiesel,50)
pacf(logDiesel,50)

decTrain<-decompose(train)
plot(decTrain$seasonal)

acf(diff(logDiesel),36)
pacf(diff(logDiesel),36)

fitArima<-arima(logDiesel,order=c(2,1,2),seasonal = c(1,1,0))
fitAutoArima<-auto.arima(train)

coeftest(fitArima)
coeftest(fitAutoArima)

qqnorm(fitArima$residuals)
qqline(fitArima$residuals)
checkresiduals(fitArima)

qqnorm(fitAutoArima$residuals)
qqline(fitAutoArima$residuals)
checkresiduals(fitAutoArima)

# Hacer el modelo
auto.arima(diesel_ts)
fit<-arima(log(diesel_ts), c(0, 1, 1),seasonal = list(order = c(0, 1, 1), period = 12))
pred<-predict(fit, n.ahead = 3)
ts.plot(diesel_ts,2.718^pred$pred, log = "y", lty = c(1,3))
fit2<-arima(log(diesel_ts), c(2, 1, 1),seasonal = list(order = c(0, 1, 0), period = 12))
forecastAP1<-forecast(fit2, level = c(95), h = 3)
autoplot(forecastAP1)

diesel_ts2018<-ts( diesel$`Diesel alto azufre`, start = c(2001,1), end=c(2020,12) ,frequency = 12)
auto.arima(diesel_ts2018)
fit<-arima(log(diesel_ts2018), c(0, 1, 1),seasonal = list(order = c(0, 1, 1), period = 12))
pred<-predict(fit, n.ahead = 3)
ts.plot(diesel_ts2018,2.718^pred$pred, log = "y", lty = c(1,3))
fit2<-arima(log(diesel_ts2018), c(2, 1, 1),seasonal = list(order = c(0, 1, 0), period = 12))
forecastAP<-forecast(fit2, level = c(95), h = 3)

autoplot(forecastAP)

df<-data.frame(ds=as.Date(as.yearmon(time(train))),y=as.matrix(train) )
testdf<-data.frame(ds=as.Date(as.yearmon(time(test))),y=as.matrix(test) )
head(df)
fitProphet<-prophet(df, yearly.seasonality = TRUE, weekly.seasonality = TRUE)
future<-make_future_dataframe(fitProphet,periods = h,freq = "month", include_history = T)
p<-predict(fitProphet,future)
p<-p[,c("ds","yhat","yhat_lower","yhat_upper")]
plot(fitProphet,p)

pred<-tail(p,h)
pred$y<-testdf$y

ggplot(pred, aes(x=ds, y=yhat)) +
  geom_line(size=1, alpha=0.8) +
  geom_ribbon(aes(ymin=yhat_lower, ymax=yhat_upper), fill="blue", alpha=0.2) +
  geom_line(data=pred, aes(x=ds, y=y),color="red")
