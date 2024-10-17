library(data.table)
library(nnet)

#Importation des données
dta <- read.table("donnees/dataset.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)
dta <- dta[c(2,3,4,6,8,9)]
summary(dta)
str(dta)

#Ajout d'une variable catégorielle pour le poids
dta$weight_factor <- cut(
  dta$weight_in_kg,
  breaks = c(150, 210, 240, 275, Inf),  # Intervalles des catégories
  labels = c("150-210", "210-240", "240-275","275+"),  # Libellés des catégories
  right = FALSE  # Inclure la borne inférieure dans chaque intervalle
)

dta <- dta[-6] # on enlève la variable quantitative weight_in_kg

#Régression
set.seed(123)  # Pour assurer la reproductibilité
trainIndex <- createDataPartition(dta$weight_factor, p = 0.7, list = FALSE)  # 70% pour le train
trainData <- dta[trainIndex,]
testData <- dta[-trainIndex,]

#Régression avec un modèle qui comprends tout
mod <- multinom(weight_factor ~ ., data = trainData)
prediction1 <- predict(object = mod, newdata = testData, type = "class")
confusionMatrix(prediction1, testData$weight_factor)

#Regression avec un modèle qui ne comprends que la taille
mod2 <- multinom(weight_factor~height_in_inch, data = trainData)
prediction2 <- predict(object = mod2, newdata = testData, type = "class")
confusionMatrix(prediction2, testData$weight_factor)

#Regression avec un modèle qui ne comprends pas la taille
testData2 <- testData[-5] #on enlève la taille des jeux de données
trainData2 <- trainData[-5] #on enlève la taille des jeux de données
mod3 <- multinom(weight_factor~., data = trainData2)
prediction3 <- predict(object = mod3, newdata = testData2, type = "class")
confusionMatrix(prediction3, testData$weight_factor)