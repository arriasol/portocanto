"""
Generem un conjunt de dades per a ciclistes en base a les categories.
Creem un fitxer CSV amb dades sobre els temps de pujada i baixada
de ciclistes, basat en un diccionari amb mitjanes i desviacions estàndard
per a diverses categories de ciclistes.

Llibreries requerides:
- os
- logging
- numpy
- pandas
- csv
"""
import os
import logging
import csv
import numpy as np
def generar_dataset(num, ind, dicc):
    """
    Genera els temps dels ciclistes, de forma aleatòria, però en base a la
    informació del diccionari i guarda les dades generades en un fitxer CSV.
    arguments:
        num -- nombre de registres de ciclistes a generar
        ind -- Index (identificador/dorsal)
        dicc -- Diccionari amb la informació de les categories
    Returns: Cap
    """

    dades_generades = []
    for i in range(num):
        categoria = np.random.choice(dicc)
        temps_pujada = int(np.random.normal(categoria["mu_p"], categoria["sigma"]))
        temps_baixada = int(np.random.normal(categoria["mu_b"], categoria["sigma"]))
        dades_generades.append({
            "dorsal": ind + i,
            "categoria": categoria["name"],
            "temps_pujada": temps_pujada,
            "temps_baixada": temps_baixada
        })

    ruta = 'data/ciclistes.csv'
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["dorsal", "categoria", "temps_pujada", "temps_baixada"])
        writer.writeheader()
        writer.writerows(dades_generades)
     # Ho afegim per fer-ho servir als testport canto ja que la primera
     # funció test_longitudddataset(self) hem donava error:
     # TypeError: object of type 'NoneType' has no len()
    return dades_generades
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    STR_CICLISTES = 'data/ciclistes.csv'
    os.makedirs(os.path.dirname(STR_CICLISTES), exist_ok=True)
    MU_P_BE = 3240
    MU_P_ME = 4268
    MU_B_BB = 1440
    MU_B_MB = 2160
    SIGMA = 240
    dicc = [
        {"name": "BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name": "MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]
    NOMBRE_CICLISTES = 150
    INDEX_INICIAL = 1
    generar_dataset(NOMBRE_CICLISTES, INDEX_INICIAL, dicc)
    logging.info("s'ha generat data/ciclistes.csv")
    