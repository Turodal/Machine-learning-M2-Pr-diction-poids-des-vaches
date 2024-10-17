library(data.table)
library(reticulate)
library(keras)
library(glue)
library(magick)
library(keras)
library(tensorflow)
library(imager)
library(dplyr)
#On utilise les features pour les img0 du jeu de données pour gagner du temps de calcul
dta <- fread("donnees/dataset.csv")
dta1 <- fread("donnees/data_img0.csv")
# dta1 <- dta1[-nrow(dta1),]

dta1 <- cbind(dta1, height_in_inch=dta$height_in_inch)

#On extrait les features et on les normalise
features <- as.matrix(dta1[,-c(1:3)])
features <- scale(features)

#On extrait les classes de poids et on les passes au format 1,2,3,4
class <- dta1[,2]
class <- as.numeric(as.factor(class$weight_factor))

num_classes <- length(unique(class))
#On passe au format 0,1,2,3 pour le réseau de neurones
labels_cat <- to_categorical(class -1, num_classes = num_classes)


#On sépare le jeu de données en train_set et test_set (80/20)
set_random_seed(245)
set.seed(245)
indices <- sample(1:nrow(dta1), size = 0.8 * nrow(dta1)) # 80% pour l'entraînement

train_features <- features[indices,]
train_label <- labels_cat[indices,]

test_features <- features[-indices,]
test_label <- labels_cat[-indices,]

colSums(train_label)
colSums(test_label)

#On définie l'architecture du modèle
model_1 <- keras_model_sequential() %>%
  layer_dense(units = 128,input_shape = c(ncol(features)))%>%
  layer_activation_leaky_relu() %>%
  layer_dropout(rate=0.5) %>% 
  layer_dense(units = 64) %>%
  layer_dropout(rate=0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

#On définie la fonction de perte et le critère d'optimisation
model_1 %>% compile(
  loss = 'categorical_crossentropy',  
  optimizer = optimizer_adam(learning_rate = 0.001), 
  metrics = c('accuracy')
)

#On entraine le modèle, les poids sont modifiés toutes les 64 lignes du train set (sélectionnées au hasard)
#Le modèle tourne comme cela 100 fois "epochs"
history <- model_1 %>% fit(
  x = train_features, 
  y = train_label,
  batch_size = 64,
  epochs = 100,
  validation_split = 0.2,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 10) #Si la valeur de le fonction de perte reste stable sur 10 epochs on arrête le l'entrainement 
  )
)

#On regarde les résultats pour le test_set
#On a une accuracy autour de 0.55
results <- model_1 %>% evaluate(test_features, test_label)
cat("Test Loss:", results["loss"], "\n")
cat("Test Accuracy:", results["accuracy"], "\n")



#Réseau de neurones pour prédire le poids d'une vache à partir des features de VGG_16----

dta1 <- fread("donnees/data_img0.csv")

#On extrait les features et les poids des vaches
features <- as.matrix(dta1[, -c(1:3)])
target <- dta1$weight_in_kg  

# Normalisation des features
features <- scale(features)

# Division des données en ensembles d'entraînement et de test
#On prédéfini une seed et on sépare les données en 80% entrainement et 20% test
set.seed(123)
set_random_seed(123)
indices <- sample(1:nrow(dta1), size = 0.8 * nrow(dta1)) # 80% pour l'entraînement

train_features <- features[indices,]
train_target <- target[indices]

test_features <- features[-indices,]
test_target <- target[-indices]

# Création du modèle : on a 3 couches
model <- keras_model_sequential() %>%
  layer_dense(units = 256,input_shape = ncol(features)) %>% 
  layer_activation_leaky_relu() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64,activation="relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)  # Couche de sortie avec 1 unité (prédiction d'une valeur continue)

#On définit la fonction de perte et le critère d'optimisation
model %>% compile(
  loss = "mse",  #La fonction de perte est l'erreur absolue moyenne
  optimizer = optimizer_adam(learning_rate = 0.008),
  metrics = c("mean_absolute_error")  # On mesure l'erreur absolue moyenne
)

# Entraînement du modèle
history <- model %>% fit(
  x = train_features,
  y = train_target,
  epochs = 300,
  batch_size = 64,
  validation_split = 0.3,  
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 10),  # Arrêt précoce si pas d'amélioration
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 5)  # Réduction du learning rate si plateau
  )
)

# Évaluation du modèle sur les données de test
results <- model %>% evaluate(test_features, test_target)
cat("Test Mean Squared Error:", results["loss"], "\n")
cat("Test Mean Absolute Error:", results["mean_absolute_error"], "\n")

# Prédiction
predictions <- model %>% predict(test_features)
print(predictions[1:10]-test_target[1:10])
print(max(predictions-test_target))
print(min(predictions-test_target))

#Le modèle actuelle a tendance à sous estimer le poids des vaches, des ajustements sont prévus
plot(test_target,predictions, xlim=c(0,900), ylim=c(-100,400))
abline(a=0,b=1)
