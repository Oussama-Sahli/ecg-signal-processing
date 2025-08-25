# ECG Signal Processing


Ceci est mon projet de traitement et d'analyse de signaux ECG, **tout dans un seul fichier Python** (`01_load_ecg.py`).  

Je vous explique ce que j’ai fait concrètement, étape par étape.

---

## 1. Objectif du projet

Le but était de **charger un segment ECG depuis PhysioNet**, filtrer le signal, détecter les battements cardiaques (R-peaks) et calculer des indicateurs de variabilité cardiaque (HRV).  

C’est un projet de **traitement de signal médical** pour comprendre le workflow complet, pas encore un vrai projet de data science prédictive.

---

## 2. Contenu du fichier unique

Le fichier `01_load_ecg.py` fait tout :

1. **Chargement du signal ECG** depuis PhysioNet (MIT-BIH).  
2. **Sauvegarde du signal brut** en CSV.  
3. **Filtrage passe-bande** entre 0.5 et 40 Hz pour enlever le bruit.  
4. **Détection des R-peaks** (battements cardiaques).  
5. **Calcul des intervalles RR**, fréquence cardiaque instantanée et moyenne.  
6. **Calcul des indicateurs HRV** : SDNN, RMSSD, NN50, pNN50.  
7. **Visualisation** : signal filtré, R-peaks, histogrammes RR et indicateurs HRV.  
8. **Simulation d’un modèle prédictif simple** de la fréquence cardiaque suivante (pour démonstration seulement).

---

## 3. Étapes réalisées

### Chargement du signal
- Téléchargement depuis PhysioNet et sauvegarde en CSV.  
- On a 10 secondes de signal avec 3600 échantillons pour chaque canal (MLII et V5).

### Filtrage du signal
- Filtre passe-bande 0.5–40 Hz pour enlever le bruit électrique et de mouvement.  
- On garde uniquement les fréquences utiles du cœur.

### Détection des R-peaks
- Chaque R-peak correspond à un battement.  
- Calcul des intervalles RR entre deux battements successifs.  

### Analyse HRV
- Calcul de la **fréquence cardiaque moyenne et instantanée**.  
- Calcul des indicateurs HRV :
  - **SDNN** : variabilité globale
  - **RMSSD** : variabilité court terme
  - **NN50** et **pNN50** : nombre et pourcentage de différences RR > 50 ms  

### Visualisation
- Graphiques du signal ECG filtré avec R-peaks.  
- Histogrammes des intervalles RR.  
- Graphiques des indicateurs HRV.

### Prédiction (démo)
- Une petite simulation pour prédire le bpm suivant à partir des intervalles RR et HRV.  
- Ce n’est pas encore un vrai projet de data science, juste pour montrer comment on pourrait utiliser un modèle.

---

## 4. Limites actuelles

- Projet **analytique**, pas un vrai modèle prédictif.  
- Il n’y a qu’un seul segment d’une personne, donc le modèle ne peut pas apprendre réellement.  
- Pour un vrai projet, il faudra :
  - Télécharger plusieurs signaux ECG de plusieurs personnes.  
  - Créer des features robustes.  
  - Prédire des événements cliniques réels comme les arythmies.

---

## 5. Comment utiliser le projet

1. Cloner le repo :  
```bash
git clone https://github.com/Oussama-Sahli/ecg-signal-processing.git
