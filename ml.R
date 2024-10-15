library(data.table)
library(caret)
library(doParallel)
library(e1071)
library(dplyr)

library(FactoMineR)
# Charger les données
img0 <- fread("data_img0.csv", stringsAsFactors = TRUE)[,-c(1,3)]

# Séparer le jeu de données en deux parties : entraînement et test
set.seed(123)  # Pour assurer la reproductibilité
trainIndex <- createDataPartition(img0$weight_factor, p = 0.7, list = FALSE)  # 70% pour le train
trainData <- dta[trainIndex,]
testData <- dta[-trainIndex,]

acp <- PCA(img0, quali.sup = 1)
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
tuneGrid <- expand.grid(mtry = c(10, 50, 60,100))


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
cm.rf <- confusionMatrix(pred, testData$weight_factor)
print(cm.rf)

importance <- varImp(mod.rf)
print(importance)
# Modèle SVM
cl <- makePSOCKcluster(detectCores() - 1)  # Réutiliser le cluster
registerDoParallel(cl)


# Prétraitement des données
preProcValues <- preProcess(trainData[, -which(names(trainData) == "weight_factor")], method = c("center", "scale"))
trainData_scaled <- predict(preProcValues, trainData)
testData_scaled <- predict(preProcValues, testData)
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
cm.svm <- confusionMatrix(pred_svm, testData_scaled$weight_factor)

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
cm.nnet <- confusionMatrix(pred.nnet, testData_scaled$weight_factor)
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
cm.knn <- confusionMatrix(pred.knn, testData_scaled$weight_factor)


#####----Comparaison entre les méthodes----#######
##################################################

acc <- list( Acc = c(cm.rf$overall[1], cm.svm$overall[1], cm.nnet$overall[1], cm.knn$overall[1]),
             Methode = c("Random Forest", "SVM", "nnet", "knn"))


plot(acc)