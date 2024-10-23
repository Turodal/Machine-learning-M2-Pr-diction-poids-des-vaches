# Machine-learning-M2
Dépot Git pour la conférence de machine learning

Résumé de la présentation : 

Le poids vif est un indicateur important pour l’élevage. Il reflète la santé, la productivité et la valeur économique des animaux. Il permet d’évaluer la croissance, d’ajuster la nutrition afin d’avoir une efficacité de nutrition maximale, de surveiller les performances reproductives, d’optimiser les stratégies de commercialisation et de choisir les animaux à abattre. La surveillance régulière du poids est essentielle pour prévenir les maladies, maximiser les gains de poids et assurer la rentabilité. Par exemple, un poids insuffisant peut indiquer des carences alimentaires, tandis qu'un excès peut signaler un risque de troubles métaboliques.

Il existe plusieurs méthodes pour mesurer le poids d’un animal, les méthodes manuelles et les méthodes automatiques. Les méthodes manuelles consistent à estimer à l’œil le poids de l’animal ou de le peser sur des balances. La première méthode n’est pas très précise et la deuxième apporte de nombreux problèmes. Les balances représentent un coût qui n’est pas à la portée de tous les éleveurs. De plus, il faut amener l’animal sur la balance, ce qui est chronophage, provoque du stress chez les animaux et peut provoquer des accidents avec des blessures sur l’animal ou l’éleveur. Grâce aux avancées technologiques, des méthodes non-invasives basées sur l'imagerie numérique sont désormais développées. Elles présentent l’avantage de minimiser le stress pour les animaux et de faciliter les mesures fréquentes dans des conditions naturelles, tout en augmentant la précision et l’efficacité dans la gestion de l’élevage et en diminuant le coût des systèmes à installer.

Pour ce sujet, nous nous basons sur un jeu de données mis en ligne par Mobasshir Bhuiya. Il comprend un tableau de données avec des informations sur 513 vaches comme la race, le poids et le sexe, ainsi que des images de ces vaches sous 4 angles différents.

Dans cette présentation, nous nous concentrons sur l’analyse d’image pour estimer le poids vif de vaches avec le réseau neuronal convolutif (CNN) VGG-16. Dans un premier temps, le réseau neuronal nous a permis d’extraire les features des différentes images pour établir un modèle linéaire. Dans un second temps, nous avons tenté de classer les vaches selon des catégories de poids en ayant entrainé VGG-16. Nous comparons le modèle linéaire avec les features à un modèle avec les informations du tableau de données. La classification par VGG-16 sera comparée avec d’autres méthodes de classification comme KNN ou Random Forest.

Mots clés : poids vif, machine learning, vache, VGG, imagerie




#Structure du dépôt Git 
Le dépôt Git est ordonné de la manière suivante
- Bibliographie : contient toutes les sources bibliographiques qui ont été étudiées 
- Données : contient les tableaux de données utilisés dans nos programmes + VGG 16 au format h5
- Image PP ML M2 : contient les images utilisés dans la présentation PPT
- Images : contient les images des vaches du jeu de données, chaque fichier contient 4 images et correspond à une vache
- Programmes : contient les différents script R utilisé, il faut d'abord lancer "Préparation environnement de travail" pour pouvoir utiliser "keras" et "tensorflow" sur R. Il peut arriver que le script ferme votre session R, dans ce cas il faut relancer. 