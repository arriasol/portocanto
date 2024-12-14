import sys
import logging
import shutil
import mlflow
import numpy as np

from sklearn.preprocessing import LabelEncoder
from mlflow.tracking import MlflowClient
sys.path.append("..")
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score

if __name__ == "__main__":
	# ##################################################
	# level=logging.INFO --> canviar entre DEBUG i INFO    
	# ##################################################
	logging.basicConfig(format='%(message)s', level=logging.INFO)

	client = MlflowClient()
	experiment_name = "K sklearn ciclistes"
	exp = client.get_experiment_by_name(experiment_name)

	if not exp:
		mlflow.create_experiment(experiment_name,
			tags={'mlflow.note.content':'portcanto variació de paràmetre K'})
		mlflow.set_experiment_tag("version", "1.0")
		mlflow.set_experiment_tag("scikit-learn", "K")
		exp = client.get_experiment_by_name(experiment_name)

	mlflow.set_experiment("K sklearn ciclistes")
    
	def get_run_dir(artifacts_uri):
		""" retorna ruta del run """
		return artifacts_uri[7:-10]

	def remove_run_dir(run_dir):
		""" elimina path amb shutil.rmtree """
		shutil.rmtree(run_dir, ignore_errors=True)

	runs = MlflowClient().search_runs(
		experiment_ids=[exp.experiment_id],
	)
	# #########################################
	# esborrem tots els runs de l'experiment
	# #########################################    
	for run in runs:
		mlflow.delete_run(run.info.run_id)
		remove_run_dir(get_run_dir(run.info.artifact_uri))    
	# #########################################        
	# Carregem i preparem el conjunt de dades.
	# #########################################    
	path_dataset = './data/ciclistes.csv'
	ciclistes_data = load_dataset(path_dataset)
	ciclistes_data = clean(ciclistes_data)
	# ##########################################################################      
	# Extreiem les etiquetes
	# Aquí hem dona error: Value erroor : setting an array element with a
	# sequence the requested arrayt has inhomogeneus shape after 1 dimensions. 
	# The deted shape was(2,) + inhomegeneous part.
	# llavors he comprovat la funció def extract_true_labels(dataframe):
	# del script clustersciclistes.py i he vist que retornava 
	# return true_labels_local, tipus_dict_local
	# ho he fet igual i ja m'ha funcionat correctament.
	# Entenc que d'aquesta manera ja hem queda un array unidemensional.    
	# ##########################################################################    
	true_labels_local, tipus_dict_local = extract_true_labels(ciclistes_data)
	true_labels = true_labels_local 
	ciclistes_data = ciclistes_data.drop('dorsal', axis=1)
	# ##########################################################################       
	# També m'ha donat el següent error: 
	# value error: could mnot convert string to float. MEBB
	# Per solucionar dit error he aplicat labelencoder.    
	# Primer de tot cerquem els columnes categoriques.
	# Després apliquem labelencoder per convertir les columnes categòriques a 
	# valor numérics. Això ho fem amb fit_transform on convertirem BEBB a 0,
	# BEMB en 1, MEBB en 2 i MEMB en 3.
	# ########################################################################## 
	categorical_columns = ciclistes_data.select_dtypes(include=['object']).columns
	if not categorical_columns.empty:
		label_encoders = {}
		for col in categorical_columns:
			le = LabelEncoder()
			ciclistes_data[col] = le.fit_transform(ciclistes_data[col])

	Ks = [2, 3, 4, 5, 6, 7, 8]
	# #########################################     
	# Per començar fem un escombrat dels valors 
	# #########################################     
	for K in Ks:
		dataset = mlflow.data.from_pandas(ciclistes_data, source=path_dataset)
		# #########################################         
		# creem el run start  i començem a fer logs.
		# #########################################         
		mlflow.start_run(description='K={}'.format(K))
		mlflow.log_input(dataset, context='training')
		# #########################################         
		# Executem el clustering "clustering_model" pel valor de K
		# que li toca des de 2 fins a 8.
		# #########################################         
		clustering_model = clustering_kmeans(ciclistes_data, K)
		data_labels = clustering_model.labels_
		# #########################################                 
		# Calculem els scores
		# #########################################                 
		h_score = round(homogeneity_score(true_labels, data_labels), 5)
		c_score = round(completeness_score(true_labels, data_labels), 5)
		v_score = round(v_measure_score(true_labels, data_labels), 5)
		# #########################################         
		# Mostrem per pantalla els scores amb logging.info        
		# #########################################         
		logging.info('K: %d', K)
		logging.info('H-measure: %.5f', h_score)
		logging.info('C-measure: %.5f', c_score)
		logging.info('V-measure: %.5f', v_score)
		# #########################################         
		# Indiquem, possem tags als nostres runs.
		# #########################################         
		tags = {
			"engineering": "AAS",
			"release.candidate": "RC1",
			"release.version": "1.1.2",
		}
		mlflow.set_tags(tags)
		# #########################################         
		# Registrem el valor de K de 2 fins a 8.
		# #########################################         
		mlflow.log_param("K", K)
		# #########################################         
		# Registrem les metriques
		# #########################################         
		mlflow.log_metric("h", h_score)
		mlflow.log_metric("c", c_score)
		mlflow.log_metric("v_score", v_score)
		# #########################################         
		# Registrem els artefactes, en el nostre 
		# cas el fitxer ciclistes.csv
		# #########################################         
		mlflow.log_artifact("./data/ciclistes.csv")
		# #########################################         
		# Parem tots els runs amb end_run()
		# #########################################         
		mlflow.end_run()        

	print('s\'han generat els runs')
