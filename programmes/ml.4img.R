library(data.table)
library(caret)
library(doParallel)
library(e1071)
library(dplyr)
library(nnet)
library(FactoMineR)
# Charger les données

dta <- fread("donnees/data_4img.csv", stringsAsFactors = TRUE, dec = ",")[,-c(1,3)]
dta <- dta[1:2052]

set.seed(123)  # Pour assurer la reproductibilité
trainIndex <- createDataPartition(dta$weight_factor, p = 0.7, list = FALSE)  # 70% pour le train
trainData <- dta[trainIndex,]
testData <- dta[-trainIndex,]

acp <- PCA(dta, quali.sup = 1)
qualitative_variable <- as.factor(dta[[1]]) 
# Définir les couleurs pour chaque niveau de la variable qualitative
colors <- c("red", "blue", "green", "orange")  # Choisissez vos 4 couleurs
names(colors) <- levels(qualitative_variable)   # Associer les couleurs aux niveaux

# Créer un vecteur de couleurs pour les individus
individual_colors <- colors[qualitative_variable]

# Tracer la PCA avec les couleurs définies
plot(acp, 
     choix = "ind",                # Choisir de tracer les individus
     col.ind = individual_colors,   # Colorer les individus selon la variable qualitative
     col.var = "black",            # Couleur des variables
     habillage = "none",           # Pas de habillage des individus
     title = "PCA avec 4 couleurs pour la variable qualitative",
     autoLab = "yes",              # Labels automatiques
     graph.type = "classic")    



# Configuration du cluster pour la parallélisation
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)  # Enregistrer le cluster

# Définir le contrôle pour la validation croisée
trainControl <- trainControl(method = "repeatedcv", number = 10, p = 0.7, repeats = 10, allowParallel = TRUE)

# Modèle Random Forest
ntree_values <- c(100, 200, 300, 400, 500) 
for (ntree in ntree_values) {
  # Entraînement du modèle Random Forest pour chaque ntree
  mod.rf <- caret::train(
    weight_factor ~ ., 
    data = trainData,
    method = "rf",
    trControl = trainControl,
    ntree = ntree,
    tuneGrid = tuneGrid
  )
  
  # Obtenir le meilleur mtry et l'accuracy associée
  best_mtry <- mod.rf$bestTune$mtry
  best_accuracy <- max(mod.rf$results$Accuracy)
  
  # Stocker les résultats dans la liste
  results[[as.character(ntree)]] <- list(
    ntree = ntree,
    best_mtry = best_mtry,
    best_accuracy = best_accuracy
  )
}
#meilleur: ntree = 500 et mtry = 100
print(results)

cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)

# Stocker les temps d'exécution
execution_times <- data.frame(Methode = character(), Time = numeric(), stringsAsFactors = FALSE)

# ---- Random Forest ----
start_time_rf <- Sys.time()

# Modèle Random Forest
tuneGrid <- expand.grid(mtry = 500)
mod.rf <- caret::train(
  weight_factor ~ ., 
  data = trainData,
  method = "rf",
  trControl = trainControl,
  ntree = 500,
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
k_values <- data.frame(k = seq(1, 100, by = 2)) 

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
k_values <- data.frame(k = 27) 

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
       subtitle = "modèles sur 4 images",
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
       subtitle = "modèles sur 4 images",
       x = "Méthode",
       y = "Temps (en secondes)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

