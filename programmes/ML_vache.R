library(data.table)
library(reticulate)
library(keras)
library(glue)
library(magick)
library(keras)
library(tensorflow)
library(imager)



dta <- fread("donnees/dataset.csv", sep = ",")
#sku est le nom de la vache
dta$sku <- as.factor(dta$sku)
#va chercher le chemin des images pour toutes les vaches
dta$img_path0 <- glue("images/{dta$sku}/{dta$sku}_0.jpg")
dta$img_path1 <- glue("images/{dta$sku}/{dta$sku}_1.jpg")
dta$img_path2 <- glue("images/{dta$sku}/{dta$sku}_2.jpg")
dta$img_path3 <- glue("images/{dta$sku}/{dta$sku}_3.jpg")



# Déterminer combien de pixels vous souhaitez enlever
nb_pixels_gauche <- 140   # Pixels à enlever de la gauche
nb_pixels_doit <- 140  # Pixels à enlever de la droite
nb_pixels_haut <- 70   # Pixels à enlever du haut
nb_pixels_bas <- 90   # Pixels à enlever du bas

#permet de centrer l'image sur la vache
centrage <- function(path) {
  image <- load.image(path)
  
  # Convertir l'image en niveaux de gris
  gray_image <- grayscale(image)
  # Vérifiez que les pixels à enlever ne dépassent pas les dimensions de l'image
  if ((nb_pixels_top + nb_pixels_bottom) >= img_dims[1]) {
    stop("Le nombre de pixels à enlever en haut et en bas est supérieur ou égal à la hauteur de l'image.")
  }
  if ((nb_pixels_left + nb_pixels_right) >= img_dims[2]) {
    stop("Le nombre de pixels à enlever à gauche et à droite est supérieur ou égal à la largeur de l'image.")
  }
  # Découper l'image pour conserver la région souhaitée
  cropped_image <- gray_image[
    (nb_pixels_gauche + 1):(img_dims[1] - nb_pixels_doit),  # Lignes
    (nb_pixels_haut + 1):(img_dims[2] - nb_pixels_bas),  # Colonnes
    drop = FALSE
  ]
  return(cropped_image)
}

#permet de segmenter l'image 
features <- function(img) {
  # Calculer le gradient de l'image
  edges <- imgradient(img, "xy")
  # Extraire les composants du gradient
  gradient_x <- edges$x
  gradient_y <- edges$y
  # Calculer l'amplitude du gradient
  magnitude_edges <- sqrt(gradient_x^2 + gradient_y^2)
  # Binariser l'amplitude des bords
  binary_edges <- magnitude_edges > 0.03  # Ajustez le seuil selon vos besoins
  # Convertir l'image binaire en matrice 2D
  binary_edges_EB <- as.Image(1-binary_edges)
  # Segmenter les objets dans l'image binaire
  segmented_edges <- bwlabel(binary_edges_EB)
  # Convertir en matrice 2D
  segmented_edges_2D <- drop(segmented_edges)
  # Extraire les caractéristiques géométriques des objets segmentés
  features_edges <- computeFeatures.shape(segmented_edges_2D)
  return(features_edges[which(features_edges ==max(features_edges)),])
}


#programme qui permet d'extraire des features simple sur la vache (air/perimètre)
polymerisation <- function(path) {
  feat <- lapply(path, FUN = function(pathe) {
    features(centrage(pathe))
  })
  poids <- data.frame(Poids = dta$weight_in_kg)
  poids_df <- do.call(rbind, feat)
  cbind(poids,poids_df)
}



# Charger VGG16 avec des poids ImageNet
weights_path <- "donnees/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Charger le modèle VGG16 sans la couche supérieure
vgg_model <- tf$keras$applications$vgg16$VGG16(weights = weights_path, include_top = FALSE)


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
  lapply(data$img_path3, function(path) {
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
dta_1img <- cell(dta)

dta_rotation <- perfect_cell(dta)
summary(dta_rotation$weight_in_kg)
c("150-210", "210-240", "240-260", "260-275", "275+")
# Définir les catégories de poids
dta_rotation$weight_factor <- cut(
  dta_rotation$weight_in_kg,
  breaks = c(150, 210, 240, 275, Inf),  # Intervalles des catégories
  labels = c("150-210", "210-240", "240-275","275+"),  # Libellés des catégories
  right = FALSE  # Inclure la borne inférieure dans chaque intervalle
)

# Définir les catégories de poids
dta_1img$weight_factor <- cut(
  dta_1img$weight_in_kg,
  breaks = c(150, 210, 240, 275, Inf),  # Intervalles des catégories
  labels = c("150-210", "210-240", "240-275","275+"),  # Libellés des catégories
  right = FALSE  # Inclure la borne inférieure dans chaque intervalle
)

features_simple <- polymerisation(dta$img_path0)
# Définir les catégories de poids
features_simple$weight_factor <- cut(
  features_simple$Poids,
  breaks = c(150, 210, 240, 275, Inf),  # Intervalles des catégories
  labels = c("150-210", "210-240", "240-275","275+"),  # Libellés des catégories
  right = FALSE  # Inclure la borne inférieure dans chaque intervalle
)
features_simple <- features_simple[, c("weight_factor", setdiff(names(features_simple), "weight_factor"))]
write.csv2(features_simple, "data_img0simple.csv")

# fait tourner les programmes pour toutes les images 
dta_1img <- dta_1img[, c("weight_factor", setdiff(names(dta_1img), "weight_factor"))]
dta_rotation <- dta_rotation[, c("weight_factor", setdiff(names(dta_rotation), "weight_factor"))]
write.csv2(dta_1img, "data_img3.csv")
write.csv2(dta_rotation, "data_img0rot.csv")

##On crée un réseau de neurones pour prédire la catégorie des vaches avec les features obtenus avec VGG----

#On utilise les features pour les img0 du jeu de données pour gagner du temps de calcul
dta1 <- fread("donnees/data_img0.csv")

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
set.seed(245)
indices <- sample(1:nrow(dta1), size = 0.8 * nrow(dta1)) # 80% pour l'entraînement

train_features <- features[indices,]
train_label <- labels_cat[indices,]

test_features <- features[-indices,]
test_label <- labels_cat[-indices,]


#On définie l'architecture du modèle
model_1 <- keras_model_sequential() %>%
  layer_dense(units = 512,input_shape = c(ncol(features)))%>%
  layer_activation_leaky_relu() %>% 
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = 256) %>%
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = num_classes, activation = 'softmax')

#On définie la fonction de perte et le critère d'optimisation
model_1 %>% compile(
  loss = 'categorical_crossentropy',  
  optimizer = optimizer_adam(learning_rate = 0.0001), 
  metrics = c('accuracy')
)

#On entraine le modèle, les poids sont modifiés toutes les 64 lignes du train set (sélectionnées au hasard)
#Le modèle tourne comme cela 100 fois "epochs"
history <- model_1 %>% fit(
  x = train_features, 
  y = train_label,
  batch_size = 64,
  epochs = 100,
  validation_data = list(test_features,test_label),
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
  layer_dense(units = 128,activation="relu") %>%
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = 128,activation="relu") %>%
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = 1)  # Couche de sortie avec 1 unité (prédiction d'une valeur continue)

#On définit la fonction de perte et le critère d'optimisation
model %>% compile(
  loss = "mse",  #La fonction de perte est l'erreur absolue moyenne
  optimizer = optimizer_adam(learning_rate = 0.0001),
  metrics = c("mean_absolute_error")  # On mesure l'erreur absolue moyenne
)

# Entraînement du modèle
history <- model %>% fit(
  x = train_features,
  y = train_target,
  epochs = 200,
  batch_size = 64,
  validation_data = list(test_features,test_target),  
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
plot(test_target,predictions)
abline(a=0,b=1)



