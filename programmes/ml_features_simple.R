library(data.table)
library(caret)
library(doParallel)
library(e1071)
library(dplyr)
library(nnet)

library(FactoMineR)

# Charger les données
dta <-  fread("donnees/data_img1simple.csv", sep = ";")[,-c(1,3,7,8,9)]

set.seed(123)  # Pour assurer la reproductibilité
trainData <- fread("donnees/data_img0simple.csv", sep = ";")[,-c(1,3,7,8,9)]
testData <- fread("donnees/data_img2simple.csv", sep = ";")[,-c(1,3,7,8,9)]
#transforme en facteur 
trainData$weight_factor <- as.factor(trainData$weight_factor)
testData$weight_factor <- as.factor(testData$weight_factor)

plot(trainData$s.area, trainData$s.perimeter, type = "p", lwd = 2, col = trainData$weight_factor)
plot(trainData$s.perimeter, trainData$s.radius.mean, type = "p", lwd = 2, col = trainData$weight_factor)


# Configuration du cluster pour la parallélisation
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)  # Enregistrer le cluster

# Définir le contrôle pour la validation croisée
trainControl <- trainControl(method = "repeatedcv", number = 10, p = 0.7, repeats = 10, allowParallel = TRUE)

# Modèle Random Forest
tuneGrid <- expand.grid(mtry = c(1,2,3,4))

mod.rf <- caret::train(
  weight_factor ~ ., 
  data = trainData,
  method = "rf",
  trControl = trainControl, # Tester différentes valeurs de `mtry`
  ntree = 100,
  tuneGrid = tuneGrid
)
stopCluster(cl)  # Arrêter le cluster
mod.rf
plot(mod.rf)
# Prédictions avec le meilleur modèle sur l'ensemble de test
pred <- predict(mod.rf, newdata = testData)
#matrice de confusion
cm.rf <- confusionMatrix(pred, testData$weight_factor)
print(cm.rf)


cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)

# Stocker les temps d'exécution
execution_times <- data.frame(Methode = character(), Time = numeric(), stringsAsFactors = FALSE)

# ---- Random Forest ----
start_time_rf <- Sys.time()

# Modèle Random Forest
tuneGrid <- expand.grid(mtry = 4)
mod.rf <- caret::train(
  weight_factor ~ ., 
  data = trainData,
  method = "rf",
  trControl = trainControl,
  ntree = 100,
  tuneGrid = tuneGrid
)

end_time_rf <- Sys.time()
execution_times <- rbind(execution_times, data.frame(Methode = "Random Forest", Time = as.numeric(difftime(end_time_rf, start_time_rf, units = "secs"))))

# Prédictions Random Forest et Matrice de confusion
pred.rf <- predict(mod.rf, newdata = testData)
cm.rf <- confusionMatrix(pred.rf, testData$weight_factor)
# Transformer la matrice en un tableau "long" (dataframe)
df_cm.rf <- as.data.frame(as.table(cm.rf))  # Convertir la matrice en dataframe "long"
colnames(df_cm.rf) <- c("Reference", "Predicted", "Count")  # Renommer les colonnes

# Créer un graphique de la matrice de confusion pour Random Forest
ggplot(data = df_cm.rf, aes(x = Reference, y = Predicted, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Count), color = "black") +  # Ajouter les nombres dans les cases
  labs(title = "Matrice de Confusion Random Forest",
       x = "Classe Réelle",
       y = "Classe Prédite") +
  theme_minimal()
# ---- SVM ----

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
#trouver les bons hyperparamètres
tuneGrid <- expand.grid(C = c(0.1, 0.01, 0.05))
mod.svm <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "svmLinear",
  trControl = trainControl,
  tuneGrid = tuneGrid
)



# Prédictions SVM et Matrice de confusion
pred.svm <- predict(mod.svm, newdata = testData_scaled)
cm.svm <- confusionMatrix(pred.svm, testData_scaled$weight_factor)

# Modèle SVM
start_time_svm <- Sys.time()
tuneGrid <- expand.grid(C = 0.01)
mod.svm <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "svmLinear",
  trControl = trainControl,
  tuneGrid = tuneGrid
)

end_time_svm <- Sys.time()
execution_times <- rbind(execution_times, data.frame(Methode = "SVM", Time = as.numeric(difftime(end_time_svm, start_time_svm, units = "secs"))))

# Prédictions SVM et Matrice de confusion
pred.svm <- predict(mod.svm, newdata = testData_scaled)
cm.svm <- confusionMatrix(pred.svm, testData_scaled$weight_factor)

# ---- glmnet ----
start_time_glmnet <- Sys.time()

# Modèle glmnet
mod.glmnet <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "glmnet",
  trControl = trainControl,
  tuneGrid = expand.grid(alpha = 0.5, lambda = 10^seq(-4, 0, length = 10)),
  metric = "Accuracy"
)

end_time_glmnet <- Sys.time()
execution_times <- rbind(execution_times, data.frame(Methode = "glmnet", Time = as.numeric(difftime(end_time_glmnet, start_time_glmnet, units = "secs"))))

# Prédictions glmnet et Matrice de confusion
pred.glmnet <- predict(mod.glmnet, newdata = testData_scaled)
cm.glmnet <- confusionMatrix(pred.glmnet, testData_scaled$weight_factor)

# ---- KNN ----
# Modèle KNN
# Renommer les niveaux manuellement  
# Utiliser make.names() pour rendre les niveaux valides
levels(trainData_scaled$weight_factor) <- make.names(levels(trainData_scaled$weight_factor))

# Faire de même pour les données de test
levels(testData_scaled$weight_factor) <- make.names(levels(testData_scaled$weight_factor))

train_control_knn <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE)
k_values <- data.frame(k = seq(1, 200, by = 2)) 

mod.knn <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "knn",
  trControl = train_control_knn,
  tuneGrid = k_values
)
plot(mod.knn)
mod.knn


# Modèle KNN
levels(trainData_scaled$weight_factor) <- make.names(levels(trainData_scaled$weight_factor))

# Faire de même pour les données de test
levels(testData_scaled$weight_factor) <- make.names(levels(testData_scaled$weight_factor))
start_time_knn <- Sys.time()
train_control_knn <- trainControl(
  method = "repeatedcv", 
  number = 10, 
  repeats = 3, 
  classProbs = TRUE, 
  allowParallel = FALSE  # Désactiver le parallélisme
)
k_values <- data.frame(k = 147) 

mod.knn <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "knn",
  trControl = train_control_knn,
  tuneGrid = k_values
)

end_time_knn <- Sys.time()
execution_times <- rbind(execution_times, data.frame(Methode = "KNN", Time = as.numeric(difftime(end_time_knn, start_time_knn, units = "secs"))))

# Prédictions KNN et Matrice de confusion
pred.knn <- predict(mod.knn, newdata = testData_scaled)
cm.knn <- confusionMatrix(pred.knn, testData_scaled$weight_factor)
cm.knn
# Arrêter le cluster
stopCluster(cl)

# ---- Afficher les matrices de confusion ----
print(cm.rf)
print(cm.svm)
print(cm.glmnet)
print(cm.knn)

# ---- Comparaison des Accuracy ----
acc_test <- list( Acc = c(cm.rf$overall[1], cm.svm$overall[1], cm.glmnet$overall[1], cm.knn$overall[1]),
                  Methode = c("Random Forest", "SVM", "glmnet", "KNN"))

acc_dftest <- data.frame(Methode = acc_test$Methode, Accuracy = acc_test$Acc)


# ---- Graphique des Accuracy sur le test ----
ggplot(acc_dftest, aes(x = Methode, y = Accuracy, fill = Methode)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Accuracy par Méthode lors du test",
       subtitle = "modèles sur features simple (image 0 et 2)",
       x = "Méthode",
       y = "Accuracy") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ---- Afficher les temps d'exécution ----
print(execution_times)

# ---- Graphique de comparaison des temps d'exécution ----
ggplot(execution_times, aes(x = Methode, y = Time, fill = Methode)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Temps d'exécution par Méthode",
       subtitle = "modèles sur features simple (image 0 et 2)",
       x = "Méthode",
       y = "Temps (en secondes)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))