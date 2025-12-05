import os
import numpy as np
import pandas as pd
import scipy.io as sio
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# 1. Configuration
root_folder = os.getcwd()
data_out_dir = os.path.join(root_folder, 'data', 'ADHD')
os.makedirs(data_out_dir, exist_ok=True)

# 2. Téléchargement
print("Téléchargement du dataset ADHD-200 (40 sujets)...")
adhd_data = datasets.fetch_adhd(n_subjects=40)
print("Téléchargement de l'atlas Harvard-Oxford...")
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

# 3. Masker
masker = NiftiLabelsMasker(labels_img=atlas.maps, standardization='zscore_sample', 
                           detrend=True, verbose=0)

# 4. Traitement
subject_ids = []
correlation_measure = ConnectivityMeasure(kind='correlation') # Mettre 'tangent' ici pour l'étape 2

print("Traitement des images (Extraction séries temporelles)...")

# --- CORRECTION CRITIQUE ICI ---
# On utilise enumerate pour aligner image et phénotype
for i, func_file in enumerate(adhd_data.func):
    try:
        # On récupère la ligne précise du DataFrame
        pheno_row = adhd_data.phenotypic.iloc[i]
        
        # On prend l'ID dans la colonne 'Subject' et on force en string
        sub_id = str(pheno_row['Subject'])
        
        print(f"Traitement sujet {sub_id} ({i+1}/{len(adhd_data.func)})...")
        
        # Création dossier
        sub_dir = os.path.join(data_out_dir, sub_id)
        os.makedirs(sub_dir, exist_ok=True)

        # Calculs
        confounds_file = adhd_data.confounds[i]
        time_series = masker.fit_transform(func_file, confounds=confounds_file)
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        
        # Sauvegarde
        mat_file = os.path.join(sub_dir, f"{sub_id}_ho_correlation.mat")
        sio.savemat(mat_file, {'connectivity': correlation_matrix})
        
        # Si tout s'est bien passé, on ajoute l'ID à la liste
        subject_ids.append(sub_id)
        
    except Exception as e:
        print(f"Erreur critique sur le sujet index {i}: {e}")

# 5. Sauvegarde Métadonnées
# On sauvegarde le CSV tel quel, il est propre
adhd_data.phenotypic.to_csv(os.path.join(data_out_dir, 'phenotypic.csv'), index=False)

# On sauvegarde la liste des IDs qui ont bien été traités
np.savetxt(os.path.join(data_out_dir, 'subject_IDs.txt'), subject_ids, fmt='%s')

print(f"Terminé ! {len(subject_ids)} sujets traités avec succès.")