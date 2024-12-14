import pickle
import pandas as pd
import logging

# ########################################################
# Primer carreguem el model
# ########################################################
with open("model/clustering_model.pkl", "rb") as f:
    clustering_model_loaded = pickle.load(f)

# ########################################################
# A continuació, carreguem el scaler
# ########################################################    
with open('model/scaler.pkl','rb') as f:
    scaler_loaded = pickle.load(f)

# ########################################################
# Finalment, carreguem el tipus_dict
# ########################################################     
with open('model/tipus_dict.pkl','rb') as f:
    tipus_loaded = pickle.load(f)

# ########################################################
# Classificació de nous valors, nova prediccció
# Per fer la predicció farem servir el dorsal,
# el temps de pujada i el temps de baixada.
# Prescidim de la última columna que es el temps
# total.
# ########################################################
nous_ciclistes = [
    [500, 3230, 1430],  # BEBB
    [501, 3300, 2120],  # BEMB
    [502, 4010, 1510],  # MEBB
    [503, 4350, 2200]   # MEMB
]

# ########################################################
# Convertim les dades a dataframe i normalitzem les dades
# ########################################################
df_nous_ciclistes = pd.DataFrame(columns=['dorsal', 'temps_pujada', 'temps_baixada'], data=nous_ciclistes)

# ########################################################
# Eliminem la columna dorsal que  no necessària abans 
# de realitzar la normalitzar
# ########################################################
df_nous_ciclistes = df_nous_ciclistes.drop(['dorsal'],axis=1)

# ########################################################
# Normalitzem les dades
# ########################################################
df_nous_ciclistes_norm = pd.DataFrame(
    scaler_loaded.transform(df_nous_ciclistes),
    index=df_nous_ciclistes.index,
    columns=df_nous_ciclistes.columns
)

# ########################################################
# Ja podem fer la prediccio, que ens retorna l'etiqueta a 
# què pertany el ciclista
# ########################################################
nova_prediccio = clustering_model_loaded.predict(df_nous_ciclistes_norm)

# ########################################################
# Finalment, a partir de l'etiqueta volem saber a quin 
# tipus pertany el ciclista:
# ########################################################

# ###############################################################
# Primer mapeguem les etiquets perque les predigui correctament.
# ja que sino hem donava error.
# ################################################################
etiquetes_tipus = {
    0: 'BEBB',
    1: 'BEMB',
    2: 'MEMB',
    3: 'MEBB'
}

for i, etiqueta in enumerate(nova_prediccio):
    # ########################################################
    # Obtenim el tipus mitjançant les etiquetes que hem
    # creat just abans.
    # ########################################################
    tipus = etiquetes_tipus.get(etiqueta)
    print(f'Ciclista {nous_ciclistes[i][0]}: Etiqueta {etiqueta}: {tipus}')
