library(data.table)
library(reticulate)
library(keras)
library(glue)
library(magick)
library(keras)
library(tensorflow)
library(imager)



dta <- fread("dataset.csv", sep = ",")
dta$sku <- as.factor(dta$sku)
dta$img_path0 <- glue("images/{dta$sku}/{dta$sku}_0.jpg")
dta$img_path1 <- glue("images/{dta$sku}/{dta$sku}_1.jpg")
dta$img_path2 <- glue("images/{dta$sku}/{dta$sku}_2.jpg")
dta$img_path3 <- glue("images/{dta$sku}/{dta$sku}_3.jpg")



# Charger VGG16 avec des poids ImageNet
weights_path <- "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Charger le modèle VGG16 sans la couche supérieure
vgg_model <- tf$keras$applications$vgg16$VGG16(weights = weights_path, include_top = FALSE)

# Vérifiez le résumé du modèle
summary(vgg_model)

# Fonction de prétraitement des images avec magick
#' Title
#'
#' @param path chemin vers l'image
#' @param angle angle selon lequel l'image est tournée
#'
#' @return
#' @export l'image sous forme de matrice, prétraite pour vgg
#'
#' @examples
preprocess_image_magick_with_rotation <- function(path, angle = 0) {
  img <- image_read(path) %>%
    image_rotate(angle) %>%  # Appliquer la rotation avec l'angle donné
    image_scale("224x224!") %>%  # Redimensionner l'image à 224x224 pixels
    image_data()  # Convertir l'image en matrice
  
  img_array <- array(as.numeric(img), dim = c(1, 224, 224, 3))  # Formater pour le modèle
  img_array <- imagenet_preprocess_input(img_array)  # Prétraiter pour VGG
  return(img_array)
}


#' Title
#'
#' @param data data avec img_path0
#' @param angle2 angle selon lequel on veut rotate l'image
#' @param boucle nombre di'iteration de la fonction 
#'
#' @return les features prédites par vgg
#' @export
#'
#' @examples
features_list <- function(data, angle2 = 0, boucle = 1) {
  lapply(data$img_path0, function(path) {
    tryCatch({
      img <- preprocess_image_magick_with_rotation(path, angle = angle2[boucle]) 
      print(boucle+10)# Utiliser la fonction de prétraitement
      features <- vgg_model(img)  # Prédire les features
      return(as.numeric(features))  # Retourner les features sous forme de vecteur
    }, error = function(e) {
      message("Erreur avec l'image : ", path, " - ", e$message)
      return(NA)  # Gérer les erreurs en renvoyant NA
    })
  })}

# Extraire les caractéristiques avec gestion d'erreurs

#' Title
#'
#' @param features_list liste des features d'une image
#' @param data data dans lequel on veut joindre les données
#'
#' @return un data avec tous les features "utiles" pour l'analyse
#' @export
#'
#' @examples
filtrer <- function(features_list, data) {
  # Filtrer les éléments non-NA
  features <- Filter(function(x) !is.na(x), features_list)
  # Concaténer les résultats si la liste n'est pas vide
  if (length(features) > 0) {
    features_df <- do.call(rbind, features)
  } else {
    features_df <- NULL  # Si tout est NA, on retourne NULL
  }
  # Supposons que 'dta' et 'features_list' sont déjà définis
  # Concaténer les données
  data <- cbind(dta, features_df)
  data <- data[, -c(1:8,10:18)]  
  # Ajouter une ligne de somme
  data[nrow(data) + 1, ] <- apply(data, 2, FUN = sum)
  last_row <- data[nrow(data), ]
  cols_to_remove <- last_row == 0
  data <- data[, !cols_to_remove, with = FALSE]
  return(data)
}


#' Title
#'
#' @param dta dta qu'on définie au début du programme
#'
#' @return un data.frame avec tous les features utiles des images 0 + le poids de l'animal 
#' @export
#'
#' @examples
cell <- function(dta) {
  total <- data.frame(Poids = dta$weight_in_kg)
  features <- features_list(dta)
  dta_filtre <- filtrer(features, dta)
  total <- cbind(total, dta_filtre)
  return(total[,-1])
}



#' Title
#'
#' @param dta dta qu'on définie au début du programme
#'
#' @returnun data.frame avec tous les features utiles des images 0 + le poids de l'animal  + les features avec 10 (mêmes) rotations
#' pour chaque image
#' @export
#'
#' @examples
perfect_cell <- function(dta) {
  total <- data.frame(Poids = dta$weight_in_kg)
  angles <- sample(c(1:360), size = 10, replace = FALSE)
  for (k in 1:10) {
    print(k)
    features <- features_list(dta, angles, boucle = k)
    print(k+100)
    dta_filtre <- filtrer(features, dta)
    print(k+1000)
    total <- cbind(total, dta_filtre)
  }
  return(total[-1])
}

set.seed(123)
rouage <- cell(dta)

golem_ancien <- perfect_cell(dta)
summary(golem_ancien$weight_in_kg)
c("150-210", "210-240", "240-260", "260-275", "275+")
# Définir les catégories de poids
golem_ancien$weight_factor <- cut(
  golem_ancien$weight_in_kg,
  breaks = c(150, 210, 240, 275, Inf),  # Intervalles des catégories
  labels = c("150-210", "210-240", "240-275","275+"),  # Libellés des catégories
  right = FALSE  # Inclure la borne inférieure dans chaque intervalle
)

rouage$weight_factor <- cut(
  rouage$weight_in_kg,
  breaks = c(150, 210, 240, 275, Inf),  # Intervalles des catégories
  labels = c("150-210", "210-240", "240-275","275+"),  # Libellés des catégories
  right = FALSE  # Inclure la borne inférieure dans chaque intervalle
)

# Vérifier les résultats
rouage <- rouage[, c("weight_factor", setdiff(names(rouage), "weight_factor"))]
golem_ancien <- golem_ancien[, c("weight_factor", setdiff(names(golem_ancien), "weight_factor"))]
write.csv2(rouage, "data_img0.csv")
write.csv2(golem_ancien, "data_img0rot.csv")


##On crée un réseau de neurones et on exploite les features extraites avec VGG----
dta1 <- fread("data_img0.csv")

rm(model_1)
features <- as.matrix(dta1[,-c(1:3)])
features <- scale(features)
class <- dta1[,2]
class <- as.numeric(as.factor(class$weight_factor))

num_classes <- length(unique(class))
labels_cat <- to_categorical(class -1, num_classes = num_classes)


set.seed(245)
indices <- sample(1:nrow(dta1), size = 0.8 * nrow(dta1)) # 80% pour l'entraînement

train_features <- features[indices,]
train_label <- labels_cat[indices,]

test_features <- features[-indices,]
test_label <- labels_cat[-indices,]


model_1 <- keras_model_sequential() %>%
  layer_dense(units = 512,input_shape = c(ncol(features)))%>%
  layer_activation_leaky_relu() %>% 
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = 256) %>%
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = num_classes, activation = 'softmax')


model_1 %>% compile(
  loss = 'categorical_crossentropy',  # Utilisé pour la classification multi-classes
  optimizer = optimizer_adam(learning_rate = 0.0001),  # Adam est un bon choix, avec un taux d'apprentissage réduit
  metrics = c('accuracy')  # Nous mesurons la précision ici
)

history <- model_1 %>% fit(
  x = train_features, 
  y = train_label,
  batch_size = 64,
  epochs = 100,
  validation_data = list(test_features,test_label),
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 10)
    )
)

results <- model_1 %>% evaluate(test_features, test_label)
cat("Test Loss:", results["loss"], "\n")
cat("Test Accuracy:", results["accuracy"], "\n")










####


dta1 <- fread("data_img0.csv")

features <- as.matrix(dta1[, -c(1:3)])  # Remplace "poids" par le nom exact de ta colonne
target <- dta1$weight_in_kg  # Variable à prédire (le poids)

# Normalisation des features
features <- scale(features)

# Division des données en ensembles d'entraînement et de test
set.seed(123)
indices <- sample(1:nrow(dta1), size = 0.8 * nrow(dta1)) # 80% pour l'entraînement

train_features <- features[indices,]
train_target <- target[indices]

test_features <- features[-indices,]
test_target <- target[-indices]

# Création du modèle
model <- keras_model_sequential() %>%
  layer_dense(units = 512,input_shape = ncol(features)) %>% 
  layer_activation_leaky_relu() %>%
  layer_dropout(rate = 0.5) %>%  # Dropout pour éviter le sur-apprentissage
  layer_dense(units = 256,activation="relu") %>% 
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = 256,activation="relu") %>%  # 2ème couche dense
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = 1)  # Couche de sortie avec 1 unité (prédiction d'une valeur continue)

# Compilation du modèle (avec MSE comme fonction de perte pour la régression)
model %>% compile(
  loss = "mse",  # Mean Squared Error pour la régression
  optimizer = optimizer_adam(learning_rate = 0.0001),  # Optimiseur Adam
  metrics = c("mean_absolute_error")  # On suit aussi l'erreur absolue moyenne (MAE)
)

# Entraînement du modèle
history <- model %>% fit(
  x = train_features,
  y = train_target,
  epochs = 100,
  batch_size = 64,
  validation_data = list(test_features,test_target),  # 20% des données d'entraînement utilisées pour la validation
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
predictions <- model %>% predict(features)
print(predictions[1:10]-target[1:10])
print(max(predictions-target))
print(min(predictions-target))


