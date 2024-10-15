library(data.table)
library(caret)
library(doParallel)
library(e1071)
library(dplyr)
library(nnet)

library(FactoMineR)

# Charger les données
dta <-  fread("donnees/data_img1simple.csv", sep = ";")[,-c(1,3)]

set.seed(123)  # Pour assurer la reproductibilité
trainData <- fread("donnees/data_img0simple.csv", sep = ";")[,-c(1,3)]
testData <- fread("donnees/data_img1simple.csv", sep = ";")[,-c(1,3)]
#transforme en facteur 
trainData$weight_factor <- as.factor(trainData$weight_factor)
testData$weight_factor <- as.factor(testData$weight_factor)

plot(trainData$s.area, trainData$s.perimeter, type = "p", lwd = 2, col = trainData$weight_factor)
plot(trainData$s.perimeter, trainData$s.radius.mean, type = "p", lwd = 2, col = trainData$weight_factor)



# Configuration du cluster pour la parallélisation
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)  # Enregistrer le cluster

# Définir le contrôle pour la validation croisée
trainControl <- trainControl(method = "repeatedcv", number = 10, p = 0.8, repeats = 10, allowParallel = TRUE)

# Modèle Random Forest
tuneGrid <- expand.grid(mtry = c(10, 50))


mod.rf <- caret::train(
  weight_factor ~ ., 
  data = trainData,
  method = "rf",
  trControl = trainControl, # Tester différentes valeurs de `mtry`
  ntree = 500,
  tuneGrid = tuneGrid
)
stopCluster(cl)  # Arrêter le cluster
mod.rf
# Prédictions avec le meilleur modèle sur l'ensemble de test
pred <- predict(mod.rf, newdata = testData)
#matrice de confusion
cm.rf <- confusionMatrix(pred, testData$weight_factor)
print(cm.rf)

class(trainData$weight_factor)
importance <- varImp(mod.rf)
print(importance)
# Modèle SVM
# Prétraitement des données

# Normaliser les colonnes (tout sauf 'weight_factor')
trainData_normalized <- trainData[, -c("weight_factor")]
# Appliquer la standardisation (centrer et réduire)
trainData_normalized <- as.data.frame(lapply(trainData_normalized, function(x) (x - mean(x)) / sd(x)))
#Ajoute le poids aux données
trainData_scaled <- cbind(weight_factor = trainData$weight_factor, trainData_normalized)


# Normaliser les colonnes (tout sauf 'weight_factor')
testData_normalized <- testData[, -c("weight_factor")]
# Appliquer la standardisation (centrer et réduire)
testData_normalized <- as.data.frame(lapply(testData_normalized, function(x) (x - mean(x)) / sd(x)))
#Ajoute le poids aux données
testData_scaled <- cbind(weight_factor = testData$weight_factor, testData_normalized)
testData_scaled[is.na(testData_scaled)] <- 0
trainData_scaled[is.na(trainData_scaled)] <- 0

#Modèle SVM
cl <- makePSOCKcluster(detectCores() - 1)  # Réutiliser le cluster
registerDoParallel(cl)
tuneGrid <- expand.grid(C = c(0.1, 1, 10))


mod.svm <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "svmLinear",
  trControl = trainControl,
  tuneGrid = tuneGrid
)

stopCluster(cl)  # Arrêter le cluster

# Prédictions sur l'ensemble de test
pred_svm <- predict(mod.svm, newdata = testData_scaled)
cm.svm <- confusionMatrix(pred_svm, testData_scaled$weight_factor)
cm.svm
trainData_scaled$weight_factor <- as.factor(trainData_scaled$weight_factor)

# Modèle NNet


cl <- makePSOCKcluster(detectCores() - 1)  # Réutiliser le cluster
registerDoParallel(cl)


mod.glm <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "multinom",
  trControl = trainControl(method = "cv", number = 10)  # Validation croisée
)

# Prédictions
pred.glm <- predict(mod.glm, newdata = testData_scaled)

# Matrice de confusion
cm.glm <- confusionMatrix(pred.glm, testData_scaled$weight_factor)
print(cm.glm)


#Transforme les facteurs pour faire les knn
levels(trainData_scaled$weight_factor) <- c("low", "medium", "high", "very_high")
levels(testData_scaled$weight_factor) <- c("low", "medium", "high", "very_high")


# Vérifiez les nouveaux niveaux
print(levels(trainData_scaled$weight_factor))
cl <- makePSOCKcluster(detectCores() - 1)  # Réutiliser le cluster
registerDoParallel(cl)

# Définir le contrôle pour la validation croisée (utiliser repeatedcv)
train_control_knn <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE)

# Définir une grille d'hyperparamètres à tester pour k
k_values <- data.frame(k = seq(1, 100, by = 2))  # Tester les valeurs de k de 1 à 20

# Entraîner le modèle KNN
mod.knn <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "knn",
  trControl = train_control_knn,
  tuneGrid = k_values
)
summary(mod.knn)
plot(mod.knn)
stopCluster(cl)  

# Prédictions NNet
pred.knn <- predict(mod.knn, newdata = testData_scaled)
cm.knn <- confusionMatrix(pred.knn, testData_scaled$weight_factor)

cm.knn
#####----Comparaison entre les méthodes----#######
##################################################

acc_test <- list( Acc = c(cm.rf$overall[1], cm.svm$overall[1], cm.glmnet$overall[1], cm.knn$overall[1]),
                  Methode = c("Random Forest", "SVM", "glm", "knn"))

acc_dftest <- data.frame(Methode = acc_test$Methode, Accuracy = acc_test$Acc)

acc_train <- list( Acc = c(mean(mod.rf$results$Accuracy), mean(mod.svm$results$Accuracy), mean(mod.glmnet$results$Accuracy), mean(mod.knn$results$Accuracy)),
                   Methode = c("Random Forest", "SVM", "glm", "knn"))
acc_dftrain <- data.frame(Methode = acc_train$Methode, Accuracy = acc_train$Acc)


# Créer le graphique à barres
ggplot(acc_dftrain, aes(x = Methode, y = Accuracy, fill = Methode)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Accuracy par Méthode lors du train",
       x = "Méthode",
       y = "Accuracy") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Créer le graphique à barres
ggplot(acc_dftest, aes(x = Methode, y = Accuracy, fill = Methode)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Accuracy par Méthode lors du test",
       x = "Méthode",
       y = "Accuracy") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

