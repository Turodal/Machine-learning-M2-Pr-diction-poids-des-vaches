library(data.table)
library(caret)
library(doParallel)
library(e1071)
library(dplyr)
library(nnet)
library(FactoMineR)
# Charger les données

dta <- img3 <- fread("donnees/data_4img.csv", stringsAsFactors = TRUE)[,-c(1,3)]
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
trainControl <- trainControl(method = "repeatedcv", number = 10, p = 0.8, repeats = 10, allowParallel = TRUE)

# Modèle Random Forest
tuneGrid <- expand.grid(mtry = seq(50, 500, by = 50))

# Liste pour stocker les résultats
results <- list()

# Boucle sur les différentes valeurs de ntree
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
#Part du principe que ntree est le bon pour tous les autres modèles


# Refais le modèle mais en concentrant les valeurs de mtry autour des valeurs que l'on suspecte être le plus elevee
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)  # Enregistrer le cluster

# Définir le contrôle pour la validation croisée
trainControl <- trainControl(method = "repeatedcv", number = 10, p = 0.8, repeats = 10, allowParallel = TRUE)

# Modèle Random Forest
tuneGrid <- expand.grid(mtry = 100)

mod.rf <- caret::train(
  weight_factor ~ ., 
  data = trainData,
  method = "rf",
  trControl = trainControl, # Tester différentes valeurs de `mtry`
  ntree = 500,
  tuneGrid = tuneGrid
)
stopCluster(cl)  # Arrêter le cluster
plot(mod.rf)
# Prédictions avec le meilleur modèle sur l'ensemble de test
pred <- predict(mod.rf, newdata = testData)
cm.rf <- confusionMatrix(pred, testData$weight_factor)
print(cm.rf)



# Modèle SVM
# Prétraitement des données

# Normaliser les colonnes (tout sauf 'weight_factor')
trainData_normalized <- trainData[, -c("weight_factor")]
# Appliquer la standardisation (centrer et réduire)
trainData_normalized <- as.data.frame(lapply(trainData_normalized, function(x) (x - mean(x)) / sd(x)))
#Ajoute le poids aux données
trainData_scaled <- cbind(weight_factor = trainData$weight_factor, trainData_normalized)
trainData_scaled[is.na(trainData_scaled)] <- 0

# Normaliser les colonnes (tout sauf 'weight_factor')
testData_normalized <- testData[, -c("weight_factor")]
# Appliquer la standardisation (centrer et réduire)
testData_normalized <- as.data.frame(lapply(testData_normalized, function(x) (x - mean(x)) / sd(x)))
#Ajoute le poids aux données
testData_scaled <- cbind(weight_factor = testData$weight_factor, testData_normalized)
testData_scaled[is.na(testData_scaled)] <- 0

#Modèle SVM
cl <- makePSOCKcluster(detectCores() - 1)  # Réutiliser le cluster
registerDoParallel(cl)
tuneGrid <- expand.grid(C = c(0, 0.01, 0.05))


mod.svm <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "svmLinear",
  trControl = trainControl,
  tuneGrid = tuneGrid
)

stopCluster(cl)  # Arrêter le cluster
plot(mod.svm)
mod.svm
# Prédictions sur l'ensemble de test
pred_svm <- predict(mod.svm, newdata = testData_scaled)
cm.svm <- confusionMatrix(pred_svm, testData_scaled$weight_factor)
cm.svm


#Choisi un C = 0.05
cl <- makePSOCKcluster(detectCores() - 1)  # Réutiliser le cluster
registerDoParallel(cl)
tuneGrid <- expand.grid(C = 0)


mod.svm <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "svmLinear",
  trControl = trainControl,
  tuneGrid = tuneGrid
)

stopCluster(cl)  # Arrêter le cluster
plot(mod.svm)
mod.svm
# Prédictions sur l'ensemble de test
pred_svm <- predict(mod.svm, newdata = testData_scaled)
cm.svm <- confusionMatrix(pred_svm, testData_scaled$weight_factor)
cm.svm


trainData_scaled$weight_factor <- as.factor(trainData_scaled$weight_factor)

# Modèle NNet


cl <- makePSOCKcluster(detectCores() - 1)  # Réutiliser le cluster
registerDoParallel(cl)


# Utiliser le modèle glmnet pour ajouter une régularisation (Elastic Net)
mod.glmnet <- train(
  weight_factor ~ ., 
  data = trainData_scaled,
  method = "glmnet",
  trControl = trainControl,
  tuneGrid = expand.grid(alpha = 0.5, lambda = 10^seq(-4, 0, length = 10)),  # Régularisation Elastic Net
  metric = "Accuracy"
)

pred.glmnt <- predict(mod.glmnet, newdata = testData_scaled)
cm.glmnet <- confusionMatrix(pred.glmnt,testData_scaled$weight_factor)
conf


levels(trainData_scaled$weight_factor) <- c("low", "medium", "high", "very_high")
levels(testData_scaled$weight_factor) <- c("low", "medium", "high", "very_high")



# Vérifiez les nouveaux niveaux
print(levels(trainData_scaled$weight_factor))
cl <- makePSOCKcluster(detectCores() - 1)  # Réutiliser le cluster
registerDoParallel(cl)

# Définir le contrôle pour la validation croisée (utiliser repeatedcv)
train_control_knn <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE)

# Définir une grille d'hyperparamètres à tester pour k
k_values <- data.frame(k = seq(1, 200, by = 2))  # Tester les valeurs de k de 1 à 20

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

acc_train <- list( Acc = c(mod.rf$results$Accuracy, mod.svm$results$Accuracy, mod.glmnet$results$Accuracy, mod.knn$results$Accuracy),
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

