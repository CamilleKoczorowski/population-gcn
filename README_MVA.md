# population-gcn
Adaptation TensorFlow 2 (compat v1) du modèle GCN pour le dataset ABIDE.

## 1. Installation

### Option A — Conda (recommandé)
```bash
conda env create -f environment.yml
conda activate population_gcn
```

### Vérification
```bash
python - << 'EOF'
import tensorflow, numpy, sklearn, nilearn
print("Environnement OK")
EOF
```

### Option B — pip (alternative)
```bash
pip install -r requirements.txt
```

## 2. Structure du repository
```
population-gcn/
 ├ gcn/               # Modèles, layers, utils GCN
 ├ data/              # Données téléchargées
 ├ notebook/          # Notebooks d'exploration
 ├ results/           # Résultats d'entraînement
 ├ ABIDEParser.py
 ├ main_ABIDE.py
 ├ train_GCN.py
 ├ environment.yml
 ├ requirements.txt
 └ README.md
```

## 3. Exécution d’un entraînement
```bash
python main_ABIDE.py --folds 11 --epochs 150 --suffix run1
```

Paramètres principaux :
- --folds : numéro du fold (0–11)
- --epochs : nombre d’epochs
- --suffix : nom du fichier de résultat sauvegardé
- --learning_rate : optionnel
- --dropout : optionnel

## 4. Dépendances principales
- numpy
- scipy
- scikit-learn
- nilearn
- nibabel
- joblib
- networkx
- matplotlib
- tqdm
- tensorflow==2.13
