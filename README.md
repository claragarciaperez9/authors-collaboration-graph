# Classification d'Articles et Réseaux de Collaboration sur arXiv
Ce projet a été réalisé par Issa Ka et Clara García dans le contexte de leur M1.

Il adopte une approche transversale pour explorer les possibilités offertes par l'analyse massive des données issues d'arXiv. Nous proposons un tour d'horizon alliant NLP (Traîtement du Langage Naturel) et Théorie des Graphes.

## NLP, Clustering et Analyse de l'Espace Vectoriel
Le projet débute par une approche de classification supervisée visant à prédire automatiquement la catégorie et la sous-catégorie scientifique d'un article à partir de son titre et de son résumé. Cette partie a été implémentée par Issa Ka et n'est pas exposé sur ce répertoire. Il a implémenté et comparé plusieurs architectures de réseaux de neurones, allant de modèles denses simples aux LSTM enrichis de mécanismes d'attention. Après il explore la structure de l'espace vectoriel (embeddings) généré dynamiquement par ces modèles afin d'en évaluer la cohérence sémantique. À l'aide de l'algorithme des K-means couplé à la réduction de dimension par UMAP, il analyse si le regroupement non supervisé des articles dans cet espace latent reproduit fidèlement la taxonomie officielle d'arXiv.

## Analyse de Réseau (Co-auteurs)
Ici on présente que la partie du graphe, une modélisation des collaborations scientifiques sous forme de **graphe pondéré non dirigé** reliant les auteurs co-signataires. Pour comprendre l'organisation de ce réseau, nous avons appliqué l'algorithme de Louvain afin d'identifier des communautés et d'analyser leurs caractéristiques communes, telles que le domaine d'expertise ou la géographie. Nous avons également interrogé la topologie du graphe pour déterminer s'il s'apparente à un « petit monde » comparable aux réseaux sociaux humains, en utilisant les calculs de plus courts chemins et de centralité intermédiaire pour évaluer la connectivité globale et détecter les chercheurs les plus influents.
