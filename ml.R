library(data.table)
library(caret)
library(doParallel)
library(e1071)

# Charger les données
img0 <- fread("data_img0.csv", stringsAsFactors = TRUE)
img0 <- img0[,-c(1,3)]

# Séparer le jeu de données en deux parties : entraînement et test
set.seed(123)  # Pour assurer la reproductibilité
trainIndex <- createDataPartition(img0$weight_factor, p = 0.7, list = FALSE)  # 70% pour le train
trainData <- img0[trainIndex,]
testData <- img0[-trainIndex,]

# Vérification de la structure des données
str(trainData)
str(testData)

# Prétraitement des données
preProcValues <- preProcess(trainData[, -which(names(trainData) == "weight_factor")], method = c("center", "scale"))
trainData_scaled <- predict(preProcValues, trainData)
testData_scaled <- predict(preProcValues, testData)

# Configuration du cluster pour la parallélisation
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)  # Enregistrer le cluster

# Définir le contrôle pour la validation croisée
trainControl <- trainControl(method = "repeatedcv", number = 10, p = 0.6, repeats = 10, allowParallel = TRUE)

# Modèle Random Forest
mtry_values <- c(6, 8, 10, 12, 16)
ntree_values <- c(100, 200, 300, 400, 500)

results_list <- list()

for (m in mtry_values) {
  for (ntree in ntree_values) {
    mod.rf <- train(
      weight_factor ~ ., 
      data = trainData,
      method = "rf",
      trControl = trainControl,
      tuneGrid = expand.grid(.mtry = m),
      ntree = ntree
    )
    results_list[[paste("mtry", m, "ntree", ntree)]] <- mod.rf
  }
}

stopCluster(cl)  # Arrêter le cluster

# Résumé des résultats
results_summary <- data.frame()
for (name in names(results_list)) {
  model <- results_list[[name]]
  results_summary <- rbind(results_summary, 
                           data.frame(
                             Model = name,
                             Accuracy = max(model$results$Accuracy),
                             Kappa = max(model$results$Kappa)
                           ))
}
print(results_summary)

# Trouver le meilleur modèle basé sur Accuracy ou Kappa
best_model_accuracy <- results_summary[which.max(results_summary$Accuracy),]
best_model_kappa <- results_summary[which.max(results_summary$Kappa),]

# Extraire le modèle correspondant dans results_list
best_model_name <- best_model_accuracy$Model
best_model <- results_list[[best_model_name]]

# Prédictions avec le meilleur modèle sur l'ensemble de test
pred <- predict(best_model, newdata = testData)
confusion_matrix <- confusionMatrix(pred, testData$weight_factor)
print(confusion_matrix)

# Modèle SVM
cl <- makePSOCKcluster(detectCores() - 1)  # Réutiliser le cluster
registerDoParallel(cl)

tuneGrid <- expand.grid(C = c(0.1, 0.5, 1, 10), 
                        sigma = c(0.01, 0.1, 0.5, 1))

mod.svm <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "svmRadial",
  trControl = trainControl,
  tuneGrid = tuneGrid
)

stopCluster(cl)  # Arrêter le cluster

# Prédictions sur l'ensemble de test
pred_svm <- predict(mod.svm, newdata = testData_scaled)
confusionMatrix(pred_svm, testData_scaled$weight_factor)

# Modèle NNet
cl <- makePSOCKcluster(detectCores() - 1)  # Réutiliser le cluster
registerDoParallel(cl)

mod.nnet <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "nnet",
  trControl = trainControl,
  trace = FALSE  # Empêcher l'affichage des détails pendant l'ajustement
)

stopCluster(cl)  # Arrêter le cluster

# Prédictions NNet
pred.nnet <- predict(mod.nnet, newdata = testData_scaled)
confusionMatrix(pred.nnet, testData_scaled$weight_factor)
levels(trainData_scaled$weight_factor) <- c("low", "medium", "high", "very_high")
levels(testData_scaled$weight_factor) <- c("low", "medium", "high", "very_high")

# Vérifiez les nouveaux niveaux
print(levels(trainData_scaled$weight_factor))
cl <- makePSOCKcluster(detectCores() - 1)  # Réutiliser le cluster
registerDoParallel(cl)

# Définir le contrôle pour la validation croisée (utiliser repeatedcv)
train_control_knn <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE)

# Définir une grille d'hyperparamètres à tester pour k
k_values <- data.frame(k = seq(1, 20, by = 2))  # Tester les valeurs de k de 1 à 20

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
confusionMatrix(pred.knn, testData_scaled$weight_factor)
