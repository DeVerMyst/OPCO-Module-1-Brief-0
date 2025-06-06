# 🚀 Notre super modèle d'iA (oui, avec un petit 'i' pour l'humilité... ou pas) 🚀

Salut à toi, explorateur du code, curieux de l'IA (ou juste là pour le café) ! Bienvenue dans le saint des saints de la prédiction, là où les bits dansent et les neurones... bah, ils calculent. On a concocté un truc pas piqué des hannetons, alors attache ta ceinture !

---

#### 🛠️ Installation : Prêt à décoller ? 🛠️

Pour que notre fusée de l'IA ne se transforme pas en vulgaire caillou, il faut un minimum de préparation.

###### Le `.venv` (alias, notre petite bulle de sérénité)

Pour éviter que ce projet ne mette le bazar dans ton PC (et vice-versa), on utilise un environnement virtuel. C'est comme une petite bulle magique où toutes les dépendances du projet vivent en harmonie, loin des conflits extérieurs.

Si tu as Python (et un peu de chance) :

```bash
python -m venv .venv
```

Puis active-le (c'est le moment "abra cadabra") :

* **Windows (PowerShell) :**
    ```bash
    .\.venv\Scripts\Activate.ps1
    ```
* **Windows (CMD) :**
    ```bash
    .\.venv\Scripts\activate.bat
    ```
* **macOS / Linux :**
    ```bash
    source .venv/bin/activate
    ```
Félicitations ! Tu es dans notre bulle. Ne respire pas trop fort, l'air y est précieux.

###### Le `requirements.txt` (alias, la liste de courses pour les geeks)

Maintenant que tu es dans la bulle, il faut la meubler avec les outils nécessaires. Ce fichier contient tout ce qu'il faut pour que Python comprenne nos blagues (et nos calculs).

Assure-toi que ton `.venv` est activé, puis :

```bash
pip install -r requirements.txt
```

Ça va mouliner un peu. C'est normal. C'est la magie qui s'opère.

---

#### 🧠 Le Modèle : Notre cerveau artificiel (enfin, un bout) 🧠

Oublie les super-ordinateurs qui prennent toute une pièce. Notre bijou, c'est un **super Neural Network (NN)** ! Oui oui, un NN à l'état de l'art (pour nous, en tout cas). Il est si avancé qu'il a :

* **2 couches "dense"** : parce que "dense", c'est le futur. Plus c'est dense, mieux c'est, n'est-ce pas ? 😉
* **1 couche de prédiction** : c'est là que la magie se produit. Elle crache la réponse, et on espère qu'elle a raison.

Prépare-toi à être émerveillé (ou juste à voir des chiffres, c'est selon).

---

#### 🗺️ Architecture : Où va quoi dans notre petit monde ? 🗺️

Pour ne pas se perdre dans les méandres de notre génie, voici comment on a organisé notre projet. C'est un peu comme une carte au trésor, mais le trésor, c'est le code !

```
.
├── data/
│   ├── df_new.csv
│   └── df_old.csv
├── models/
│   ├── models.py
│   ├── model_2024_08.pkl
│   └── preprocessor.pkl
├── modules/
│   ├── evaluate.py
│   ├── preprocess.py
│   └── print_draw.py
├── .gitignore
├── README.md
├── main.py
└── requirements.txt
```

###### `data/` (Le garde-manger du projet)
Ici, c'est là que nos précieuses données vivent.
* `df_new.csv` : Les données fraîches du jour, prêtes à être dévorées par notre IA.
* `df_old.csv` : Les classiques, les vétérans, ceux qui ont tout vu. On les garde par nostalgie (et pour la rétrospective).

###### `models/` (Le garage à cerveaux)
Ce dossier, c'est notre caverne d'Ali Baba des cerveaux artificiels.
* `models.py` : Les plans de nos futurs cyborgs... euh, de nos modèles. C'est ici que l'on définit l'architecture de nos NN et autres merveilles.
* `model_2024_08.pkl` : Une version sauvegardée de notre modèle. On l'a encapsulé pour qu'il ne s'échappe pas et ne domine pas le monde... pas encore.
* `preprocessor.pkl` : L'outil magique qui prépare les données avant de les donner à manger au modèle. Sans lui, c'est l'indigestion assurée !

###### `modules/` (La boîte à outils de MacGyver)
Ce sont nos couteaux suisses du code. Chaque fichier est un expert dans son domaine.
* `evaluate.py` : Le juge impitoyable qui dit si notre modèle est un génie ou un cancre.
* `preprocess.py` : Le chef cuisinier des données. Il les nettoie, les coupe, les assaisonne pour qu'elles soient parfaites pour notre IA.
* `print_draw.py` : L'artiste du groupe. Il transforme nos chiffres barbares en beaux graphiques pour que même ta grand-mère puisse comprendre (enfin, presque).

---

On espère que cette petite virée dans notre projet t'a plu. N'hésite pas à jeter un œil au `main.py` pour lancer le grand spectacle !

*Fait avec amour, code et une bonne dose de caféine (et un peu de folie).*


# TD => GOGOGO
## setup

- ### Génération requirements.txt à chaque installation de module
```bash
pip freeze > requirements.txt
```

- ### Installations des requis loguru: 
```bash
pip install loguru
```

- ### Installations des requis FastAPI/Streamlit: 
```bash
pip install nltk fastapi streamlit uvicorn requests pydantic
```
- #### Pour lancer le serveur MLflow :
```bash
uvicorn mlFlow_experiment:app --host 127.0.0.1 --port 9000 --reload
```
- #### Description des routes de l'API FastAPI :
[GET /docs](http://127.0.0.1:9000/docs#/)


- ### Installation des bibliothèques pour les tests unitaires: 
```bash
pip install pytest httpx
pytest test_predict_api.py
```

- ### Installations des requis pour MLflow : 
  > **mlFlow**
  MlFlow est un outil de gestion des expériences de machine learning. Il permet de suivre les expériences, de gérer les modèles et de visualiser les résultats.
  

```bash
pip install mlflow scikit-learn pandas matplotlib
```

# Pour lancer le serveur MLflow :
```bash
mlflow ui
```

Création d'un script pour générer 3 entrainements et les stocker sur MLflow : 
Les models créé sont stockés dans le dossier `models/` et pictures du drawloss sont stockés dans le dossier `figures/`.


```code

# Question pour follow-up :

- C'est quoi la meilleur valeur ? 
- 5 entrainnements ce n'est pas assez ? 
- influence du random ?

- Quel interet de réentrainer si on a déja entrainé un modèle jusqu'à sa meilleur valeur?

- X = df.drop(columns=["nom", "prenom", "montant_pret"])
=> "est ce qu'on retire nom premon parce que ça n'a pas de sens statistique ou pour éthique et rgpd ?"



# Entrainements RESULTS
## Entrainement sur les **anciennes données** :
==================Performance for run 0/5===================
MSE: 35648332.4621, MAE: 4868.7535, R²: 0.7571
============================================================
==================Performance for run 1/5===================
MSE: 21110141.2394, MAE: 3514.9460, R²: 0.8562
============================================================
==================Performance for run 2/5=================== => *Best OLD*
MSE: 21087705.3786, MAE: 3499.9798, R²: 0.8563
============================================================
==================Performance for run 3/5===================
MSE: 21079447.9325, MAE: 3500.0007, R²: 0.8564
============================================================
==================Performance for run 4/5===================
MSE: 21088151.5108, MAE: 3503.9029, R²: 0.8563
============================================================

***

## Train de 5 avec les nouvelles données : 
==================Performance for run 0/5===================
MSE: 17590206.6978, MAE: 2934.2774, R²: 0.8375
============================================================
==================Performance for run 1/5===================
MSE: 13534161.0819, MAE: 2393.7978, R²: 0.8750
============================================================
==================Performance for run 2/5=================== => *Best NEW*
MSE: 13484831.7345, MAE: 2363.3283, R²: 0.8754
============================================================
==================Performance for run 3/5===================
MSE: 13548757.8251, MAE: 2348.1724, R²: 0.8748
============================================================
==================Performance for run 4/5===================
MSE: 13529502.2254, MAE: 2339.3840, R²: 0.8750
============================================================

***

## 3 entrainement depuis le meilleur modèle de la passe old (60ca87cde38a42dab673c4c6491ba076) avec les données new pour voir si on a un meilleur résultat :
==================Performance for run 0/3===================
MSE: 13519659.5319, MAE: 2393.4214, R²: 0.8751
============================================================
==================Performance for run 1/3===================
MSE: 13564161.1756, MAE: 2389.5547, R²: 0.8747
============================================================
==================Performance for run 2/3===================
MSE: 13500678.8155, MAE: 2363.6760, R²: 0.8753
============================================================


```
> Arrivé au bout de la 3ème itération, on a un modèle qui semble stable et performant sur les nouvelles données. 
> **Pas d'amélioration notable** de réentrainer sur les ancienne données





