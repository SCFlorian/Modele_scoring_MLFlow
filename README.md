# Modèle de scoring

L'objectif de ce projet est de réaliser un modèle de scoring et un suivi sur MlFlow à partir du projet Home Credit Default Risk sur Kaggle.

### Problématique

- Construire et optimiser un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à la pré-production du modèle.

### Données utilisées

- Les différents fichiers viennent du projet Kaggle :

https://www.kaggle.com/c/home-credit-default-risk/data

- Nous aurons un fichier global fusionné avant de pouvoir modéliser.

### Organisation du projet
```
├── Data
│   ├── Raw/          # Fichiers à ajouter car trop lourd pour GitHub
│   └── Processed/    # Fichiers à ajouter car trop lourd pour GitHub
│
├── mlruns            # Dossiers à ajouter après avoir cloné le projet
├── notebooks/        # Étapes du projet sous forme de notebooks
│   ├── notebook_1_analyse_exploratoire.ipynb
│   ├── notebook_2_feature_engineering.ipynb
│   ├── notebook_3_modele_classification.ipynb
│
├── src/        
│   ├── build_dataset.py
│   ├── data_cleaning.py
│   ├── feature_aggregations.py
│   ├── impute_numeric_only.py
│   ├── prepare_application_data.py
│ 
├── .gitignore
├── README.md         # Documentation du projet
└── pyproject.toml    # Dépendances et configuration
```
### Installation et utilisation
1. Cloner le projet :
``` 
git clone https://github.com/SCFlorian/Modele_scoring_MLFlow
cd Modele_scoring_MLFlow
```
2. Installer les dépendances :
Le projet utilise pyproject.toml pour la gestion des dépendances.
```
poetry install
```
3. Ouvrir le projet dans VS Code
```
code .
```
4. Configurer l’environnement Python dans VS Code
	1.	Installez l’extension Python (si ce n’est pas déjà fait).
	2.	Appuyez sur Ctrl+Shift+P (Windows/Linux) ou Cmd+Shift+P (Mac).
	4.	Recherchez “Python: Select Interpreter”.
	5.	Sélectionnez l’environnement créé par Poetry ou celui dans lequel tu as installé le projet.