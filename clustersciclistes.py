"""
Analitzem el comportament de ciclistes utilitzant clustering KMeans.
Aquest script proporciona funcions per carregar dades, netejar-les, aplicar clustering,
i generar informes amb visualitzacions gràfiques.
Llibreries requerides:
- os
- logging
- pickle
- pandas
- seaborn
- matplotlib
- sklearn
"""
import os
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from sklearn import preprocessing

@contextmanager
def suppress_stdout_stderr():
    """
    Gestor de context per suprimir la sortida estàndard i d'error.
    Utilitzem aquest gestor per evitar que es mostrin missatges a la consola
    mentre s'executa una acció dins d'aquest context.
    """
    with open(os.devnull, 'w', encoding='utf-8') as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            yield
def load_dataset(path):
    """
    Carrega un conjunt de dades des d'un fitxer CSV a un DataFrame de pandas.
    Arguments:
        path (str): Ruta al fitxer CSV.
    Return:
        pandas.DataFrame: Conjunt de dades carregat.
    Exemple:
        >>> load_dataset('data/ciclistes.csv')
    """
    return pd.read_csv(path, delimiter=',')
def eda(dataframe):
    """
    Realitza una anàlisi exploratòria del conjunt de dades (EDA).
    Arguments:
        dataframe (pandas.DataFrame): Conjunt de dades a analitzar.
    Returns:
        cap
    """
    logging.debug('\n%s', dataframe.shape)
    logging.debug('\n%s', dataframe.head(4))
    logging.debug('\n%s', dataframe.columns)
    logging.debug('\n%s', dataframe.info())
def clean(dataframe):
    """
    Netegem el conjunt de dades eliminant columnes innecessàries.
    Arguments:
        dataframe (pandas.DataFrame): Conjunt de dades a netejar.
    Returns:
        pandas.DataFrame: Conjunt de dades netejat.
    Exemple:
        >>> clean(dataframe)
    """
    dataframe = dataframe.drop(['id', 'tt'], axis=1, errors='ignore')
    logging.debug('\nDataframe:\n%s\n...', dataframe.head(3))
    return dataframe
def extract_true_labels(dataframe):
    """
    Extreu les etiquetes de veritat (labels) del conjunt de dades.
    Arguments:
        dataframe(pandas.DataFrame): Conjunt de dades amb la columna 'categoria'.
    Returns:
        tupla: (numpy.ndarray de les etiquetes de veritat, diccionari de categories a índex).
    Excepcions:
        KeyError: Si la columna 'categoria' no existeix.
    Exemple:
        >>> extract_true_labels(dataframe)
    """
    if 'categoria' not in dataframe.columns:
        raise KeyError("Error: La columna 'categoria' no existeix. Reviseu les dades.")
    true_labels_local = dataframe['categoria']
    tipus_dict_local = {label: idx for idx, label in enumerate(true_labels_local.unique())}
    return true_labels_local, tipus_dict_local
def associar_clusters_patrons(tipus_cluster, model):
    """
    Associa les etiquetes dels clusters amb patrons de comportament.
    Arguments:
        tipus (llista de diccionaris): Tipus de comportaments a associar amb els clusters.
        model (KMeans): Model KMeans entrenat.
    Return:
        llista de diccionaris: Tipus actualitzats amb les etiquetes dels clusters.
    """
    dicc = {'tp': 0, 'tb': 1}
    logging.info('Centres dels clusters:')
    for j, center in enumerate(model.cluster_centers_):
        logging.info('%d: (tp: %.1f, tb: %.1f)', j, center[dicc['tp']], center[dicc['tb']])
    ind_label_0, ind_label_3 = -1, -1
    suma_max, suma_min = 0, float('inf')
    for j, center in enumerate(model.cluster_centers_):
        suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
        if suma > suma_max:
            suma_max, ind_label_3 = suma, j
        if suma < suma_min:
            suma_min, ind_label_0 = suma, j
    tipus_cluster[0]['label'], tipus_cluster[3]['label'] = ind_label_0, ind_label_3
    remaining_labels = set(range(len(model.cluster_centers_))) - {ind_label_0, ind_label_3}
    sorted_labels = sorted(remaining_labels, key=lambda x: model.cluster_centers_[x][dicc['tp']])
    tipus_cluster[1]['label'], tipus_cluster[2]['label'] = sorted_labels
    logging.info('Tipus actualitzats i etiquetes: %s', tipus_cluster)
    return tipus_cluster
def normalitzacio(dataframe, escalar_model):
    """
    Normalitzem el conjunt de dades utilitzant un escalador pre-entrenat.
    Arguments:
        dataframe (pandas.DataFrame): Conjunt de dades a normalitzar.
        escalar_model (preprocessing.StandardScaler): Escalador pre-entrenat.
    Return:
        pandas.DataFrame: Conjunt de dades normalitzat.
    """
    return pd.DataFrame(escalar_model.transform(dataframe), columns=dataframe.columns)
def visualitzar_pairplot(dataframe):
    """
    Genera un gràfic pairplot per visualitzar les correlacions entre atributs.
    Arguments:
        dataframe (pandas.DataFrame): Conjunt de dades a visualitzar.
    Returns:
        cap
    """
    sns.pairplot(dataframe)
    os.makedirs('img', exist_ok=True)
    plt.savefig("img/pairplot.png")
    logging.info("Pairplot generat i guardat a img/pairplot.png")
def clustering_kmeans(data, n_clusters=4):
    """
    Aplica el mètode de clustering KMeans sobre les dades.
    Arguments:
        data (pandas.DataFrame): Dades d'entrada per al clustering.
        n_clusters (int): Nombre de clusters (per defecte 4).
    Returns:
        KMeans: Model KMeans entrenat.
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    with suppress_stdout_stderr():
        model.fit(data)
    logging.info('Centres dels clusters: %s', model.cluster_centers_)
    return model
def generar_informes(data, tipus):
    """
    Genera informes generals i específics per cada cluster.
    Arguments:
        data (pandas.DataFrame): Dades amb les etiquetes dels clusters.
        tipus (llista de diccionaris): Tipus de clusters i etiquetes associades.
    Returns:
        cap
    """
    os.makedirs('informes', exist_ok=True)
    with open('informes/tots_els_clusters.txt', 'w', encoding='utf-8') as escriure_file:
        escriure_file.write('--- Informe General ---\n')
        escriure_file.write(data.head().to_string() + '\n')
        escriure_file.write('--- Tipus de Clusters ---\n')
        escriure_file.write(str(tipus) + '\n')
    for label in sorted(data['label'].unique()):
        cluster_data = data[data['label'] == label]
        cluster_type = next((t['name'] for t in tipus if t['label'] == label), f'Cluster {label}')
        with open(f'informes/cluster_{label}.txt', 'w', encoding='utf-8') as escriure_file:
            escriure_file.write(f'--- Informe del Cluster {label} ---\n')
            escriure_file.write(f'Tipus: {cluster_type}\n')
            escriure_file.write(cluster_data.to_string(index=False))
def visualitzar_clusters(data, labels):
    """
    Visualitza els clusters amb colors diferents.
    Arguments:
        data (pandas.DataFrame): Dades d'entrada.
        labels (llista o array): Etiquetes dels clusters.
    Returns:
        cap
    """
    os.makedirs('img', exist_ok=True)
    sns.scatterplot(x='temps_pujada', y='temps_baixada', data=data, hue=labels, palette="rainbow")
    plt.savefig("img/clusters.png")
    logging.info("Visualització dels clusters guardada a img/clusters.png")
def predir_noves_dades(noves_dades, scaler, model):
    """
    Prediu les etiquetes dels clusters per noves dades.
    Arguments:
        noves_dades (pandas.DataFrame): Noves dades a classificar.
        scaler (preprocessing.StandardScaler): Escalador pre-entrenat.
        model (KMeans): Model KMeans entrenat.
    Retorna:
        pandas.DataFrame: Noves dades amb etiquetes predites.
    """
    noves_dades_norm = pd.DataFrame(scaler.transform(noves_dades), columns=noves_dades.columns)
    noves_dades_norm['label'] = model.predict(noves_dades_norm)
    return noves_dades_norm
if __name__ == "__main__":
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    PATH_DATASET = './data/ciclistes.csv'
    ciclistes_data = load_dataset(PATH_DATASET)
    eda(ciclistes_data)
    ciclistes_data = clean(ciclistes_data)
    logging.info('Columnes disponibles: %s', ciclistes_data.columns)
    true_labels, tipus_dict = extract_true_labels(ciclistes_data)
    tipus_dict = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]
    columnes_numeriques = ['temps_pujada', 'temps_baixada']
    ciclistes_data = ciclistes_data[columnes_numeriques]
    scaler_model = preprocessing.StandardScaler().fit(ciclistes_data)
    ciclistes_data_norm = normalitzacio(ciclistes_data, scaler_model)
    clustering_model = clustering_kmeans(ciclistes_data_norm)
    tipus_associar = associar_clusters_patrons(tipus_dict, clustering_model)
    ciclistes_data_norm['label'] = clustering_model.labels_
    data_labels = clustering_model.labels_
    generar_informes(ciclistes_data_norm, tipus_associar)
    os.makedirs('model', exist_ok=True)
    try:
        with open('model/tipus_dict.pkl', 'wb') as sortida_file:
            pickle.dump(tipus_dict, sortida_file)
    except IOError as e:
        logging.error("Error en guardar tipus_dict.pkl: %s", e)
    try:
        with open('model/scaler.pkl', 'wb') as sortida_file:
            pickle.dump(scaler_model, sortida_file)
    except IOError as e:
        logging.error("Error en guardar scaler.pkl: %s", e)
    ciclistes_data_norm = normalitzacio(ciclistes_data, scaler_model)
    try:
        with open('model/clustering_model.pkl', 'wb') as sortida_file:
            pickle.dump(clustering_model, sortida_file)
    except IOError as e:
        logging.error("Error en guardar clustering_model.pkl: %s", e)
    visualitzar_pairplot(ciclistes_data_norm)
    visualitzar_clusters(ciclistes_data_norm, true_labels)
    scores = {
        'homogeneity': homogeneity_score(true_labels, data_labels),
        'completeness': completeness_score(true_labels, data_labels),
        'v_measure': v_measure_score(true_labels, data_labels),
    }
    try:
        with open('model/score.pkl', 'wb') as sortida_file:
            pickle.dump(scores, sortida_file)
    except IOError as e:
        logging.error("Error en guardar score.pkl: %s", e)
    logging.info('\nHomogeneity: %.3f', scores['homogeneity'])
    logging.info('Completeness: %.3f', scores['completeness'])
    logging.info('V-Measure: %.3f', scores['v_measure'])
    nous_ciclistes = pd.DataFrame(
    [
        [500, 3230],   # BEBB
        [3300, 2120],  # BEMB
        [4010, 1510],  # MEBB
        [4350, 2200]   # MEMB
    ],
    columns=['temps_pujada', 'temps_baixada']
    )
    prediccions = predir_noves_dades(nous_ciclistes, scaler_model, clustering_model)
    for i, p in enumerate(prediccions['label']):
        t = [t for t in tipus_associar if t['label'] == p]
        if not t:
            logging.warning('No s\'ha trobat cap tipus associat per al cluster %s', p)
            continue
        logging.info('tipus %s (%s) - classe %s', i, t[0]['name'], p)
