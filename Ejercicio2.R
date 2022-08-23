#Librerias
library("readxl")
library(dplyr)
library(forecast)
library(keras)
library(tensorflow)
library(ggplot2)
library(recipes)
library(lubridate)

#Leer xls files
consumo<-read_excel("Consumo.xlsx")

consumo$Fecha<-as.Date(consumo$Fecha, "%Y/%m/%d")
str(consumo)

#Serie de tiempo con LSTM
#Consumo de Diesel
fecha<-consumo[,'Fecha']
diesel<-consumo[,'Diesel alto azufre']
dieselc<-consumo[c('Fecha','Diesel alto azufre')]

diesel_ts<-ts(dieselc$`Diesel alto azufre`, start = c(2001,1),frequency = 12)

#Gráfico de la serie a utilizar
serie<-diff(diesel_ts) #Quitar esto para ver la predicción sin estacionarizar la serie en media
plot(serie)

#Normalizar la serie
serie_norm<-scale(serie)

#Transformar a una serie supervisada
lagged<-c(rep(NA,1),serie_norm[1:(length(serie_norm)-1)])
supervisada<-as.data.frame(cbind(lagged,serie_norm))
colnames(supervisada)<-c("x-1","x")
supervisada[is.na(supervisada)]<-0

#Cantidad de elementos de cada conjunto
entrenamiento<-round(0.6*length(serie))
val_prueba<-round(0.2*length(serie))

#El test son los últimos
test<-tail(supervisada,val_prueba)
#Se corta la matriz
supervisada<-supervisada %>% head(nrow(.)-val_prueba)
#Se saca el conjunto de validación y se corta nuevamente
validation<-supervisada %>% tail(val_prueba)
supervisada<-head(supervisada,nrow(supervisada)-val_prueba)
#El train son los que quedan
train<-supervisada
rm(supervisada)

#Division en entrenamiento, prueba y validación
y_train<-train[,2]
x_train<-train[,1]
y_val<-validation[,2]
x_val<-validation[,1]
y_test<-test[,2]
x_test<-test[,1]

#Convertir a matrices
paso <- 1
caracteristicas<-1 #es univariada
dim(x_train) <- c(length(x_train),paso,caracteristicas)
dim(y_train) <- c(length(y_train),caracteristicas)
dim(x_test) <- c(length(x_test),paso,caracteristicas)
dim(y_test) <- c(length(y_test),caracteristicas)
dim(x_val) <- c(length(x_val),paso,caracteristicas)
dim(y_val) <- c(length(y_val),caracteristicas)

#Creando el modelo
lote = 1
unidades<-1
modelo1<-keras_model_sequential()
modelo1 %>% 
  layer_lstm(unidades, batch_input_shape=c(lote,paso,caracteristicas),
             stateful = T) %>%
  layer_dense(units = 1)

summary(modelo1)

#Compilar el modelo
modelo1 %>%
  compile(
    optimizer = "rmsprop",
    loss = "mse"
  )

#Entrenar el modelo
epocas <- 50
history <- modelo1 %>% fit(
  x = x_train,
  y = y_train,
  validation_data = list(x_val, y_val),
  batch_size = lote,
  epochs = epocas,
  shuffle = FALSE,
  verbose = 0
)

#Graficar el modelo
plot(history)

#Evaluar el modelo
print("Entrenamiento")
modelo1 %>% evaluate(
  x = x_train,
  y = y_train
)
print("Validación")
modelo1 %>% evaluate(
  x = x_val,
  y = y_val
)
print("Prueba")
modelo1 %>% evaluate(
  x = x_test,
  y = y_test
)

#Predicción modelo 1
