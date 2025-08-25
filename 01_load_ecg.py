# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
01_load_ecg.py

Mon objectif :
--------------
Charger un petit segment d'ECG depuis la base MIT-BIH (PhysioNet),
regarder les métadonnées de base et sauvegarder un échantillon
au format CSV pour pouvoir le réutiliser plus tard.

C’est la première étape de mon projet : CHARGEMENT DES DONNÉES.
"""

# =====================
# 1) J'importe les librairies dont j'ai besoin
# =====================
import wfdb          # pour récupérer les signaux ECG depuis PhysioNet
import numpy as np   # pour manipuler mes vecteurs/matrices
import pandas as pd  # pour exporter les données en CSV
from pathlib import Path  # pour gérer les dossiers proprement

# =====================
# 2) Je définis mes paramètres
# =====================
RECORD_ID = "100"       # identifiant de l'enregistrement ECG dans MIT-BIH
DURATION_SECONDS = 10   # durée du segment que je veux charger (en secondes)

# Je crée les dossiers pour sauvegarder les données
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)  # je crée le dossier s'il n'existe pas

# =====================
# 3) Je définis une fonction pour charger le signal ECG
# =====================
def load_mitbih_segment(record_id: str, duration_seconds: int):
    """
    Cette fonction me permet de télécharger et charger un segment d'ECG
    depuis PhysioNet (MIT-BIH).

    Paramètres
    ----------
    record_id : str
        Identifiant de l'enregistrement (par exemple "100").
    duration_seconds : int
        Durée du segment que je veux (en secondes).

    Retour
    ------
    t : np.ndarray
        Le vecteur temps en secondes.
    ecg : np.ndarray
        Le signal ECG (matrice N x C où C = nb de canaux).
    fs : float
        La fréquence d'échantillonnage (Hz).
    meta : dict
        Un dictionnaire avec des infos utiles (noms des canaux, unités, etc.).
    """
    # Je lis d'abord l'en-tête pour connaître la fréquence d'échantillonnage
    header = wfdb.rdheader(record_id, pn_dir="mitdb")
    fs = header.fs

    # Je calcule combien d'échantillons correspondent à la durée voulue
    sampto = int(duration_seconds * fs)

    # Je lis le signal ECG sur la durée choisie
    rec = wfdb.rdrecord(record_id, pn_dir="mitdb", sampto=sampto)
    ecg = rec.p_signal  # ça me donne un tableau (N, C)

    # Je construis mon vecteur temps
    N = ecg.shape[0]
    t = np.arange(N) / fs

    # Je récupère des infos utiles (métadonnées)
    meta = {
        "record_id": record_id,
        "fs": fs,
        "n_samples": N,
        "n_channels": ecg.shape[1],
        "sig_name": rec.sig_name,
        "units": getattr(rec, "units", ["mV"] * ecg.shape[1]),
    }
    return t, ecg, fs, meta

import os  # pour gérer les dossiers

# =====================
# 4) Je lance le programme principal
# =====================
if __name__ == "__main__":
    print("\n=== Chargement d'un segment ECG depuis PhysioNet (MIT-BIH) ===")
    try:
        # J'appelle ma fonction pour charger le signal
        t, ecg, fs, meta = load_mitbih_segment(RECORD_ID, DURATION_SECONDS)

        # J’affiche quelques infos de base
        print(f"Record: {meta['record_id']} | Fs: {meta['fs']} Hz")
        print(f"Nb d'échantillons: {meta['n_samples']} | Nb de canaux: {meta['n_channels']}")
        print(f"Canaux: {meta['sig_name']} | Unités: {meta['units']}")

        # Je mets mes données dans un DataFrame pour mieux les manipuler
        df = pd.DataFrame({"time_s": t})
        for ch_idx, ch_name in enumerate(meta["sig_name"]):
            df[f"ecg_{ch_name}"] = ecg[:, ch_idx]

        # Je définis le chemin de sauvegarde
        save_dir = "C:/Users/osahl/Documents/Git_hub_Oussama/ecg-signal-processing/data"
        # Je crée le dossier s'il n'existe pas
        os.makedirs(save_dir, exist_ok=True)

        # Je sauvegarde le CSV
        out_csv = os.path.join(save_dir, f"mitdb_{RECORD_ID}_{DURATION_SECONDS}s.csv")
        df.to_csv(out_csv, index=False)

        print(f"\n Fichier sauvegardé : {out_csv}")
        print(f"Durée approx.: {df['time_s'].iloc[-1]:.2f} s")

    except Exception as e:
        print("\n Erreur lors du chargement :", repr(e))
        print("Vérifie ta connexion Internet et que la librairie wfdb est bien installée.")







#----------------------------------------------------------------------------
# === Affichage du signal ECG brut ===

import pandas as pd
import matplotlib.pyplot as plt

# Je récupère le fichier CSV que j'ai sauvegardé dans mon projet GitHub
file_path = "C:/Users/osahl/Documents/Git_hub_Oussama/ecg-signal-processing/data/mitdb_100_10s.csv"
df = pd.read_csv(file_path)

# Je regarde les premières lignes pour comprendre ce que j'ai
print(df.head())

# On va afficher les deux canaux : MLII et V5
# Attention, dans mon CSV j'ai nommé les colonnes "ecg_MLII" et "ecg_V5"
plt.figure(figsize=(12, 6))
plt.plot(df['ecg_MLII'], label='MLII')
plt.plot(df['ecg_V5'], label='V5')
plt.title("Signal ECG brut sur 10 secondes")
plt.xlabel("Échantillons (Fs = 360 Hz)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.show()




#----------------------------------------------------------------------------
# === prétraitement du signal ECG ===


import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

file_path = "C:/Users/osahl/data/raw/mitdb_100_10s.csv"
df = pd.read_csv(file_path)
print(df.head())  # Je vérifie que mes colonnes MLII et V5 sont bien là



"""
Ici, j’utilise scipy.signal pour filtrer mon signal.
 On veut enlever le bruit qui peut venir des mouvements ou du 50/60 Hz du secteur électrique.
"""



# === Définir un filtre passe‑bande pour l’ECG ===


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=0.5, highcut=40, fs=360):
    b, a = butter_bandpass(lowcut, highcut, fs)
    y = filtfilt(b, a, data)
    return y



"""
Ici je filtre entre 0.5 Hz et 40 Hz, ce qui correspond aux fréquences utiles du cœur.
Tout ce qui est en dehors (bruit de mouvement, secteur électrique) sera supprimé.
"""



# === Appliquer le filtre sur mes deux canaux ===

fs = 360  # fréquence d'échantillonnage
df['MLII_filt'] = apply_bandpass_filter(df['ecg_MLII'], fs=fs)
df['V5_filt'] = apply_bandpass_filter(df['ecg_V5'], fs=fs)




# === Afficher le signal filtré ===
plt.figure(figsize=(12, 6))
plt.plot(df['MLII_filt'], label='MLII filtré')
plt.plot(df['V5_filt'], label='V5 filtré')
plt.title("Signal ECG filtré sur 10 secondes")
plt.xlabel("Échantillons (Fs = 360 Hz)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.show()




#----------------------------------------------------------------------------
# === détecter les battements du cœur (R-peaks) ===


# === Détection des R-peaks ===

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# === 1) Je charge mon signal filtré ===
file_path = "C:/Users/osahl/data/raw/mitdb_100_10s.csv"
df = pd.read_csv(file_path)

# Pour l'instant, je prends directement le signal MLII brut
# Plus tard je pourrai mettre le signal filtré
df['MLII_filt'] = df['ecg_MLII']

# === 2) Détection des R-peaks ===
# Je veux repérer les pics R qui correspondent aux battements de cœur
# height=0.3 : je ne veux garder que les pics significatifs, pas le bruit
# distance=200 : je ne veux pas compter deux pics trop proches comme deux battements
peaks, _ = find_peaks(df['MLII_filt'], height=0.3, distance=200)

# Je regarde combien de battements j'ai détecté
print(f"Nombre de battements détectés : {len(peaks)}")

# === 3) Affichage du signal avec les R-peaks ===
# Je veux voir visuellement si les R-peaks sont bien placés
plt.figure(figsize=(12, 6))
plt.plot(df['MLII_filt'], label='MLII filtré')
plt.plot(peaks, df['MLII_filt'][peaks], 'rx', label='R-peaks')  # 'rx' = croix rouge sur les pics
plt.title("Détection des R-peaks sur le signal MLII")
plt.xlabel("Échantillons (Fs = 360 Hz)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.show()

"""
Nombre de battements détectés : 13
Donc le cœur bat environ 78 fois par minute sur ce segment, ce qui est normal pour un adulte au repos
"""



#----------------------------------------------------------------------------
# === Calculer les intervalles RR ===


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# === Je charge le fichier filtré que j'ai sauvegardé ===
file_path = "C:/Users/osahl/data/raw/mitdb_100_10s.csv"
df = pd.read_csv(file_path)

# === Je récupère le signal filtré MLII ===
# Ici je suppose que j'ai déjà appliqué le filtre et sauvegardé MLII_filt
signal = df['ecg_MLII'].values

# === Détection des R-peaks ===
# Je fixe une hauteur minimale pour éviter les petits bruits
peaks, _ = find_peaks(signal, distance=200, height=0.5)  # distance en échantillons
print(f"Nombre de battements détectés : {len(peaks)}")

# === Je calcule les intervalles RR ===
# Intervalle entre deux R-peaks consécutifs (en échantillons)
rr_intervals = np.diff(peaks)  # en nombre d'échantillons
# Convertir en secondes
fs = 360  # fréquence d'échantillonnage
rr_intervals_s = rr_intervals / fs

# === Je fais un petit résumé ===
print("\n=== Statistiques des intervalles RR ===")
print(f"Moyenne RR : {np.mean(rr_intervals_s):.3f} s")
print(f"Écart-type RR : {np.std(rr_intervals_s):.3f} s")

# === Affichage du signal avec R-peaks ===
plt.figure(figsize=(12, 6))
plt.plot(signal, label='MLII filtré')
plt.plot(peaks, signal[peaks], 'rx', label='R-peaks')
plt.title("Signal ECG filtré avec R-peaks détectés")
plt.xlabel("Échantillons")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.show()


# === Résumé de la variabilité cardiaque ===
# La moyenne des intervalles RR nous donne la fréquence cardiaque moyenne
# L'écart-type montre la régularité des battements
# Ici, HR moyenne ~ 74 bpm (60 / 0.806)

"""
Moyenne RR = 0.806 s → ça veut dire que, en moyenne, le cœur bat toutes les 0,806 secondes.

Écart-type RR = 0.072 s → ça nous indique la variabilité du rythme cardiaque :
plus c’est petit, plus le cœur bat régulièrement.
"""


hr_mean = 60 / np.mean(rr_intervals_s)
print(f"Fréquence cardiaque moyenne : {hr_mean:.1f} bpm")

# Fréquence cardiaque moyenne : 74.4 bpm 



#------------------------------------------------------------------------------
# analyse HRV

import matplotlib.pyplot as plt
import numpy as np

# === Je récupère les intervalles RR en secondes ===
# rr_intervals_s vient de la détection des R-peaks
# On a déjà fait : rr_intervals_s = np.diff(peaks) / fs

# === Histogramme des intervalles RR ===
plt.figure(figsize=(8,5))
plt.hist(rr_intervals_s, bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution des intervalles RR")
plt.xlabel("Intervalle RR (s)")
plt.ylabel("Nombre d'occurrences")
plt.show()

# === Calcul de la fréquence cardiaque instantanée ===
hr_inst = 60 / rr_intervals_s  # en bpm
print("\nFréquence cardiaque instantanée (bpm) :")
print(hr_inst)

# === Affichage de la fréquence cardiaque au fil du temps ===
plt.figure(figsize=(10,5))
plt.plot(hr_inst, marker='o', linestyle='-')
plt.title("Fréquence cardiaque instantanée")
plt.xlabel("Battements successifs")
plt.ylabel("Fréquence (bpm)")
plt.grid(True)
plt.show()




#------------------------------------------------------------------------------
# tableau récapitulatif des statistiques de notre ECG



import pandas as pd
import numpy as np

# === Je récupère les intervalles RR et la fréquence cardiaque instantanée ===
# rr_intervals_s et hr_inst viennent de la détection des R-peaks
# On les a calculés précédemment

# === Calcul des statistiques simples ===
stats_dict = {
    "Moyenne RR (s)": np.mean(rr_intervals_s),
    "Écart-type RR (s)": np.std(rr_intervals_s),
    "Fréquence cardiaque moyenne (bpm)": 60 / np.mean(rr_intervals_s),
    "Fréquence cardiaque max (bpm)": np.max(hr_inst),
    "Fréquence cardiaque min (bpm)": np.min(hr_inst)
}

# === Création d'un DataFrame pour un affichage propre ===
df_stats = pd.DataFrame(stats_dict, index=[0])

# === Affichage ===
print("\n=== Statistiques du segment ECG ===")
print(df_stats)

# === Option : sauvegarde pour GitHub ===
out_csv = "C:/Users/osahl/Documents/Git_hub_Oussama/ecg-signal-processing/data/ecg_stats_segment.csv"
df_stats.to_csv(out_csv, index=False)
print(f"\nFichier sauvegardé : {out_csv}")



#------------------------------------------------------------------------------
# indicateurs de variabilité cardiaque (HRV) sur notre segment ECG


import numpy as np
import pandas as pd

# === Je récupère les intervalles RR déjà calculés en secondes ===
# rr_intervals_s vient de la détection des R-peaks
# On a déjà fait : rr_intervals_s = np.diff(peaks) / fs

# === SDNN : écart-type des intervalles RR ===
sdnn = np.std(rr_intervals_s)
print(f"SDNN (variabilité globale) : {sdnn:.3f} s")

# === RMSSD : variabilité à court terme ===
rmssd = np.sqrt(np.mean(np.diff(rr_intervals_s)**2))
print(f"RMSSD (variabilité court terme) : {rmssd:.3f} s")

# === NN50 et pNN50 ===
nn50 = np.sum(np.abs(np.diff(rr_intervals_s)) > 0.05)  # > 50 ms
pnn50 = nn50 / len(rr_intervals_s) * 100
print(f"NN50 : {nn50}, pNN50 : {pnn50:.1f} %")

# === On met tout ça dans un DataFrame pour sauvegarder ===
hrv_metrics = pd.DataFrame({
    "SDNN_s": [sdnn],
    "RMSSD_s": [rmssd],
    "NN50": [nn50],
    "pNN50_%": [pnn50]
})

# === On sauvegarde le résultat ===
hrv_metrics.to_csv("C:/Users/osahl/Documents/Git_hub_Oussama/ecg-signal-processing/data/ecg_hrv_metrics.csv", index=False)
print("Indicateurs HRV sauvegardés !")


"""

SDNN (variabilité globale) : 0.072 s
RMSSD (variabilité court terme) : 0.123 s
NN50 : 3, pNN50 : 25.0 %
Indicateurs HRV sauvegardés !


SDNN = 0.072 s → la variabilité globale est assez faible (cœur régulier sur ces 10 s).

RMSSD = 0.123 s → la variabilité à court terme est un peu plus importante, ce qui est normal pour les battements proches.

NN50 = 3 et pNN50 = 25 % → 25 % des intervalles RR ont une différence supérieure à 50 ms, c’est une info sur la variabilité ponctuelle.



"""


#------------------------------------------------------------------------------
#  visualisation et l’analyse des tendances du cœur




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Je récupère mes intervalles RR et la fréquence cardiaque instantanée ===
# rr_intervals_s vient de la détection des R-peaks
# hr_inst = 60 / rr_intervals_s  (bpm)

# Pour l’exemple, je simule les données (remplace par tes variables)
rr_intervals_s = np.array([0.806,0.806,0.76,0.76,0.76,0.734,0.656,0.995,0.842,0.813,0.792,0.771])
hr_inst = 60 / rr_intervals_s

# === 1) Histogramme des intervalles RR ===
plt.figure(figsize=(8,5))
plt.hist(rr_intervals_s, bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution des intervalles RR")
plt.xlabel("Intervalle RR (s)")
plt.ylabel("Nombre d'occurrences")
plt.show()
# -> Ici je peux voir si mes battements sont réguliers ou irréguliers

# === 2) Fréquence cardiaque instantanée au fil des battements ===
plt.figure(figsize=(10,5))
plt.plot(hr_inst, marker='o', linestyle='-')
plt.title("Fréquence cardiaque instantanée")
plt.xlabel("Battements successifs")
plt.ylabel("Fréquence (bpm)")
plt.grid(True)
plt.show()
# -> Ça me montre comment le cœur varie d’un battement à l’autre

# === 3) Boxplot des intervalles RR ===
plt.figure(figsize=(6,4))
plt.boxplot(rr_intervals_s, vert=True, patch_artist=True)
plt.title("Boxplot des intervalles RR")
plt.ylabel("Intervalle RR (s)")
plt.show()
# -> Utile pour voir la variabilité globale et détecter des valeurs aberrantes




#------------------------------------------------------------------------------
#  indices de variabilité cardiaque (HRV)


import numpy as np
import pandas as pd

# === Je récupère mes intervalles RR en secondes ===
# rr_intervals_s vient de la détection des R-peaks
# Pour l’exemple, on reprend nos données
rr_intervals_s = np.array([0.806,0.806,0.76,0.76,0.76,0.734,0.656,0.995,0.842,0.813,0.792,0.771])

# === 1) SDNN : écart-type global des intervalles RR ===
sdnn = np.std(rr_intervals_s)
print(f"SDNN (variabilité globale) : {sdnn:.3f} s")

# === 2) RMSSD : racine carrée de la moyenne des différences au carré (variabilité court terme) ===
diff_rr = np.diff(rr_intervals_s)
rmssd = np.sqrt(np.mean(diff_rr**2))
print(f"RMSSD (variabilité court terme) : {rmssd:.3f} s")

# === 3) NN50 et pNN50 : nombre de paires d'intervalles dont la différence > 50 ms ===
nn50 = np.sum(np.abs(diff_rr) > 0.05)  # 50 ms = 0.05 s
pnn50 = 100 * nn50 / len(diff_rr)
print(f"NN50 : {nn50}, pNN50 : {pnn50:.1f} %")

# === 4) Sauvegarder les indicateurs HRV dans un CSV ===
hrv_dict = {
    "SDNN_s": [sdnn],
    "RMSSD_s": [rmssd],
    "NN50": [nn50],
    "pNN50_pct": [pnn50]
}
df_hrv = pd.DataFrame(hrv_dict)
df_hrv.to_csv("C:/Users/osahl/Documents/Git_hub_Oussama/ecg-signal-processing/data/hrv_indicators.csv", index=False)
print("Indicateurs HRV sauvegardés !")



"""
SDNN = 0.076 s → le rythme cardiaque est relativement stable sur ces 10 secondes.

RMSSD = 0.116 s → il y a un peu de variation rapide entre certains battements, ce qui est normal.

NN50 = 3, pNN50 = 27.3 % → 3 intervalles ont une différence > 50 ms par rapport au précédent, soit 27 % des battements.
"""



#------------------------------------------------------------------------------
#  variabilité de mon rythme cardiaque (HRV) de manière claire


import pandas as pd
import matplotlib.pyplot as plt

# === Je charge mes statistiques HRV sauvegardées ===
file_path = "C:/Users/osahl/Documents/Git_hub_Oussama/ecg-signal-processing/data/ecg_hrv_segment.csv"
df = pd.read_csv(file_path)

# === Je regarde ce que j'ai ===
print(df.head())

# === Histogramme SDNN ===
# SDNN = variabilité globale des intervalles RR
plt.figure(figsize=(8,5))
plt.bar(["SDNN"], df["SDNN"], color="orange")
plt.title("Variabilité globale (SDNN)")
plt.ylabel("Secondes")
plt.show()

# === Histogramme RMSSD ===
# RMSSD = variabilité court terme
plt.figure(figsize=(8,5))
plt.bar(["RMSSD"], df["RMSSD"], color="green")
plt.title("Variabilité court terme (RMSSD)")
plt.ylabel("Secondes")
plt.show()

# === Diagramme NN50 et pNN50 ===
plt.figure(figsize=(8,5))
plt.bar(["NN50", "pNN50"], [df["NN50"][0], df["pNN50"][0]], color=["blue","purple"])
plt.title("Nombre et pourcentage de différences RR > 50 ms")
plt.ylabel("Valeur")
plt.show()




#------------------------------------------------------------------------------
# démo  prédiction machine learning


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ==========================
# 1) Je simule des données ECG pour plusieurs "personnes"
# ==========================
np.random.seed(42)  # pour reproduire les résultats

n_samples = 50  # nombre de segments simulés
# Moyenne RR autour de 0.8 s, léger bruit
moy_rr = 0.8 + 0.05 * np.random.randn(n_samples)
# Écart-type RR autour de 0.07 s
std_rr = 0.07 + 0.02 * np.random.randn(n_samples)
# Fréquence cardiaque moyenne
hr_mean = 60 / moy_rr + np.random.randn(n_samples)  # bpm
# HRV globale (SDNN)
sdnn = std_rr
# HRV court terme (RMSSD)
rmssd = 0.12 + 0.03 * np.random.randn(n_samples)

# Je mets tout dans un DataFrame
data = pd.DataFrame({
    "Moyenne RR (s)": moy_rr,
    "Écart-type RR (s)": std_rr,
    "Fréquence cardiaque moyenne (bpm)": hr_mean,
    "SDNN": sdnn,
    "RMSSD": rmssd
})

# Target : fréquence cardiaque suivante simulée
hr_next = hr_mean + 2 * np.random.randn(n_samples)  # variation légère
data["HR_next"] = hr_next

print("Aperçu des données simulées :")
print(data.head())

# ==========================
# 2) Je prépare mes features et ma target
# ==========================
X = data[["Moyenne RR (s)", "Écart-type RR (s)", "Fréquence cardiaque moyenne (bpm)", "SDNN", "RMSSD"]].values
y = data["HR_next"].values

# ==========================
# 3) Split train/test
# ==========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================
# 4) Modèle RandomForest
# ==========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================
# 5) Prédictions
# ==========================
y_pred = model.predict(X_test)

# ==========================
# 6) Performance
# ==========================
mse = mean_squared_error(y_test, y_pred)
print(f"Erreur quadratique moyenne : {mse:.3f}")

# ==========================
# 7) Comparaison visuelle
# ==========================
plt.figure(figsize=(10,5))
plt.plot(y_test, marker='o', label='Vrai HR')
plt.plot(y_pred, marker='x', label='Prédit HR')
plt.title("Prédiction de la fréquence cardiaque (données simulées)")
plt.xlabel("Segments test")
plt.ylabel("Fréquence cardiaque (bpm)")
plt.legend()
plt.grid(True)
plt.show()


"""
j’ai créé un petit modèle Random Forest qui essaie de prédire 
la fréquence cardiaque du prochain battement à partir de la moyenne RR, 
de l’écart-type RR et de la fréquence cardiaque moyenne.
"""



