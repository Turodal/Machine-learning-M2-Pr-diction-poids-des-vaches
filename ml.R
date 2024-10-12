library(data.table)
library(caret)
library(doParallel)
img0 <- fread("data_img0.csv", stringsAsFactors = TRUE)
img0 <- img0[,-c(1,3)]

# Séparer le jeu de données en deux parties : entraînement et test
set.seed(123)  # Pour assurer la reproductibilité
trainIndex <- createDataPartition(img0$weight_factor, p = 0.7, list = FALSE)  # 70% pour le train
# Jeu de données d'entraînement et de test
trainData <- img0[trainIndex,]
testData <- img0[-trainIndex,]

cl <- makePSOCKcluster(detectCores() - 1)  # Utiliser tous les cœurs sauf un
registerDoParallel(cl)  # Enregistrer le cluster

# Définir le contrôle pour la validation croisée
trainControl <- trainControl(method = "repeatedcv", number = 10, p = 0.6, repeats = 10, allowParallel = TRUE)
# Définir une grille d'hyperparamètres à tester
mtry_values <- c(6, 8, 10, 12, 16)
ntree_values <- c(100, 200, 300, 400, 500)

# Initialiser une liste pour stocker les résultats
results_list <- list()

# Boucle pour tester chaque combinaison de mtry et ntree
for (m in mtry_values) {
  for (ntree in ntree_values) {
    # Entraîner le modèle Random Forest
    mod.rf <- train(
      weight_factor ~ ., 
      data = trainData,
      method = "rf",
      trControl = trainControl,
      tuneGrid = expand.grid(.mtry = m),
      ntree = ntree,  # Spécifier ntree ici
      allowParallel = TRUE
    )
    
    # Stocker les résultats
    results_list[[paste("mtry", m, "ntree", ntree)]] <- mod.rf
  }
}
print(results_list)
stopCluster(cl)
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

best_model_name <- best_model_accuracy$Model

# Extraire le modèle correspondant dans results_list
best_model <- results_list[[best_model_name]]

# Faire des prédictions avec le meilleur modèle sur l'ensemble de test
pred <- predict(best_model, newdata = testData)

# Afficher les résultats des prédictions
print(pred)

# Évaluer les performances avec une matrice de confusion
confusion_matrix <- confusionMatrix(pred, testData$weight_factor)
print(confusion_matrix)

# Faire des prédictions sur l'ensemble de test
pred <- predict(best_model_accuracy, newdata = testData)

# Matrice de confusion
confusionMatrix(pred, testData$weight_factor)

# Entraîner le modèle Random Forest sur l'ensemble d'entraînement
mod.rfopt <- train(
  weight_factor ~ ., 
  data = trainData,
  method = "rf",
  trControl = trainControl,
  tuneGrid = expand.grid(.mtry = 10),  # Pas nécessaire si mtry est déjà défini
  ntree = 500  # Spécifier ntree ici
)

pred.opt <- predict(mod.rfopt, newdata = testData)
confusionMatrix(pred.opt, testData$weight_factor)

library(e1071)
preProcValues <- preProcess(trainData, method = c("center", "scale"))
trainData_scaled <- predict(preProcValues, trainData)
testData_scaled <- predict(preProcValues, testData)

cl <- makePSOCKcluster(detectCores() - 1)  # Utiliser tous les cœurs sauf un
registerDoParallel(cl)  # Enregistrer le cluster
tuneGrid <- expand.grid(C = c(0.1, 0.5, 1, 10))
mod.svm <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "svmLinear",
  trControl = trainControl,
  tuneGrid = tuneGrid
)
stopCluster(cl)
plot(mod.svm)
summary(mod.svm)
pred_svm <- predict(mod.svm, newdata = testData_scaled)
confusionMatrix(pred_svm, testData_scaled$weight_factor)


# Configuration du cluster pour la parallélisation
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)  # Enregistrer le cluster

# Définir une grille d'hyperparamètres à tester pour le SVM
tuneGrid <- expand.grid(C = c(0.1, 0.5, 1, 10), 
                        sigma = c(0.01, 0.1, 0.5, 1))
# Entraîner le modèle SVM avec la recherche de grille
mod.svm <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "svmRadial",  # Utiliser un noyau radial pour plus de flexibilité
  trControl = trainControl,
  tuneGrid = tuneGrid
)
plot(mod.svm)
# Arrêter le cluster
stopCluster(cl)

# Faire des prédictions sur l'ensemble de test
pred_svm <- predict(mod.svm, newdata = testData_scaled)

# Évaluer les performances avec une matrice de confusion
confusionMatrix(pred_svm, testData_scaled$weight_factor)



cl <- makePSOCKcluster(detectCores() - 1)  # Utiliser tous les cœurs sauf un
registerDoParallel(cl)  # Enregistrer le cluster

mod.nnet <- train(
  weight_factor ~ ., 
  data = trainData,
  method = "nnet",
  trControl = trainControl
)
stopCluster(cl)

summary(mod.nnet)
pred.nnet <- predict(mod.nnet, newdata = testData)
confusionMatrix(pred.nnet, testData$weight_factor)
