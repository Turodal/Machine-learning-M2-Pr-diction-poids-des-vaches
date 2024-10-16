library(data.table)
library(reticulate)
library(keras)
library(glue)
library(magick)
library(keras)
library(tensorflow)
library(imager)
library(dplyr)



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
features_list <- function(data, angle2 = 0, boucle = 1, chemin) {
  lapply(data[[chemin]], function(path) {
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


super_polymerisation <- function(features_list, data) {
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

true_filtrage <- function(data) {
  data[nrow(data) + 1, ] <- apply(data, 2, FUN = sum)
  last_row <- data[nrow(data), ]
  cols_to_remove <- last_row == 0
  data <- data[, !cols_to_remove]
  return(data)
}



four_sword <- function(dta) {
  total <- data.frame(Poids = dta$weight_in_kg)
  images <- c("img_path0","img_path1","img_path2","img_path3")
  for (k in 1:4) {
    poids <- data.frame(Poids = dta$weight_in_kg)
    print(images[k])
    features <- features_list(dta, chemin = images[k])
    dta_fusionne <- super_polymerisation(features)
    print(k+100)
    print(k+1000)
    if (k == 1){
      print(dim(total))
      print(dim(dta_fusionne))
      total <- cbind(total, dta_fusionne)
    } else {
      poids <- cbind(poids, dta_fusionne)
      total <- rbind(poids,total)
    }
    
  }
  return(total[-1])
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
  angles <- c(0,90,180,270)
  for (k in 1:4) {
    poids <- data.frame(Poids = dta$weight_in_kg)
    print(k)
    features <- features_list(dta$img0, angles, boucle = k)
    dta_fusionne <- super_polymerisation(features)
    print(k+100)
    print(k+1000)
    if (k == 1){
      print(dim(total))
      print(dim(dta_fusionne))
      total <- cbind(total, dta_fusionne)
    } else {
      poids <- cbind(poids, dta_fusionne)
      total <- rbind(poids,total)
    }
    
  }
  return(total[-1])
}

set.seed(123)
dta_1img <- cell(dta)



dta_rotation <- perfect_cell(dta)
dta_rotation <- true_filtrage(dta_rotation)

dta_4img <- four_sword(dta)
dta_4img <- true_filtrage(dta_4img)

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

dta_4img$weight_in_kg <- as.numeric(dta_4img$weight_factor)

dta_4img$weight_factor <- cut(
  dta_4img$weight_in_kg,
  breaks = c(150, 210, 240, 275, Inf),  # Intervalles des catégories
  labels = c("150-210", "210-240", "240-275","275+"),  # Libellés des catégories
  right = FALSE  # Inclure la borne inférieure dans chaque intervalle
)
dta_4img <- dta_4img[, c("weight_factor", setdiff(names(dta_4img), "weight_factor"))]

# fait tourner les programmes pour toutes les images 
dta_1img <- dta_1img[, c("weight_factor", setdiff(names(dta_1img), "weight_factor"))]
dta_rotation <- dta_rotation[, c("weight_factor", setdiff(names(dta_rotation), "weight_factor"))]
write.csv2(dta_1img, "data_img3.csv")
write.csv2(dta_rotation, "data_img0rot.csv")
write.csv2(dta_4img, "data_4img.csv")

