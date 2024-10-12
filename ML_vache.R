library(data.table)
library(reticulate)
library(keras)
library(glue)
library(magick)
library(keras)
library(tensorflow)
library(imager)
# Vérifiez quel Python est utilisé


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

read.csv2(rouage, "data_img0")
read.csv2(golem_ancien, "data_img0rot")