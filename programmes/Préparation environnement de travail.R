#Ligne de code pour installer keras et Tensorflow sur R

#Installation de remotes pour installer tensorflow
install.packages("remotes")
remotes::install_github("rstudio/tensorflow")


#à activer si python n'est pas installé sur l'ordinateur
remotes::install_python(version="3.10.11")

#Création d'un environnement python virtuel pour pouvoir utiliser tensorflow
library(tensorflow)
remotes::install_tensorflow(envname = "r-tensorflow")

#Installation de keras 
remotes::install.packages("keras")
library(keras)
install_keras()

#Ces lignes de codes peuvent prendre du temps pour se lancer mais son nécéssaire pour pouvoir lancer les autres programmes