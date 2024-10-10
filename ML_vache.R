library(data.table)
library(reticulate)
library(keras)
library(glue)
library(magick)
library(keras)
library(tensorflow)
library(imager)
# Vérifiez quel Python est utilisé
py_config()
py_install("Pillow")
dta <- fread("dataset.csv", sep = ",")
dta$sku <- as.factor(dta$sku)
dta$img_path0 <- glue("images/{dta$sku}/{dta$sku}_0.jpg")
dta$img_path1 <- glue("images/{dta$sku}/{dta$sku}_1.jpg")
dta$img_path2 <- glue("images/{dta$sku}/{dta$sku}_2.jpg")
dta$img_path3 <- glue("images/{dta$sku}/{dta$sku}_3.jpg")


tensorflow::install_tensorflow()

library(tensorflow)
# Installer Keras
keras::install_keras()
model <- application_ssd_mobilenet_v2()
# Charger VGG16 avec des poids ImageNet
weights_path <- "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Charger le modèle VGG16 sans la couche supérieure
vgg_model <- tf$keras$applications$vgg16$VGG16(weights = weights_path, include_top = FALSE)

# Vérifiez le résumé du modèle
summary(vgg_model)

# Fonction de prétraitement des images avec magick
preprocess_image_magick <- function(path) {
  img <- image_read(path) %>%
    image_scale("224x224!") %>%  # Redimensionner l'image à 224x224 pixels
    image_data()  # Convertir l'image en matrice
  img_array <- array(as.numeric(img), dim = c(1, 224, 224, 3))  # Formater pour le modèle
  img_array <- imagenet_preprocess_input(img_array)  # Prétraiter pour VGG
  return(img_array)
}


# Exemple d'utilisation pour extraire les features
features_list <- lapply(dta$img_path0, function(path) {
  tryCatch({
    img <- preprocess_image_magick(path)  # Utiliser la fonction de prétraitement
    features <- vgg_model(img) # Prédire les features
    return(as.numeric(features))
  }, error = function(e) {
    message("Erreur avec l'image : ", path, " - ", e$message)
    return(NA)  # Gérer les erreurs
  })
})

all(is.na(features_list))

library(data.table)
library(caret)

# Supposons que 'dta' et 'features_list' sont déjà définis

# Concaténer les données
features_df <- do.call(rbind, features_list)
data <- cbind(dta, features_df)
data <- data[, c(9, 19:25106)]  # Ajustez les indices selon votre structure

# Ajouter une ligne de somme
data[nrow(data) + 1, ] <- apply(data, 2, FUN = sum)

# Vérifier les valeurs de la dernière ligne et supprimer les colonnes avec des zéros
last_row <- data[nrow(data), ]
cols_to_remove <- last_row == 0
data <- data[, !cols_to_remove, with = FALSE]

# Entraîner le modèle avec validation croisée
control <- trainControl(method = "cv", number = 10)
model <- train(weight_in_kg ~ ., data = data, method = "lm", trControl = control)

# Prédire les poids
predicted_weights <- predict(model, newdata = features_df)

# Afficher les poids prédits
print(predicted_weights)


library(moments)
img <- load.image(dta$img_path2[1])
gray_img <- grayscale(img)

# Aplatir l'image en vecteur
img_vector <- as.numeric(gray_img)

# Calculer les moments d'image (par exemple, moment de Skewness)
skewness_value <- skewness(img_vector)

# Afficher les moments
print(skewness_value)
