import unittest
import os
import pickle
import numpy as np

from generardataset import generar_dataset
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score

class TestGenerarDataset(unittest.TestCase):
	"""
	classe TestGenerarDataset
	"""
	global mu_p_be
	global mu_p_me
	global mu_b_bb
	global mu_b_mb
	global sigma
	global dicc

	mu_p_be = 3240 # mitjana temps pujada bons escaladors
	mu_p_me = 4268 # mitjana temps pujada mals escaladors
	mu_b_bb = 1440 # mitjana temps baixada bons baixadors
	mu_b_mb = 2160 # mitjana temps baixada mals baixadors
	sigma = 240 # 240 s = 4 min

	dicc = [
		{"name":"BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
		{"name":"MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
	]
	def test_longituddataset(self):
		"""
		Test la longitud de l'array
		"""
		nombre_ciclistes = 150
		index = 0
		# Li indiquem el diccionari amb els valors de les constants
		# entenc que ho havia d'agafar class TestGenerarDataset
		# però al final ho he fet així        
		dicc = [
			{"name": "BEBB", "mu_p": 3240, "mu_b": 1440, "sigma": 240},
			{"name": "BEMB", "mu_p": 3240, "mu_b": 2160, "sigma": 240},
			{"name": "MEBB", "mu_p": 4268, "mu_b": 1440, "sigma": 240},
			{"name": "MEMB", "mu_p": 4268, "mu_b": 2160, "sigma": 240}
		]
		# Passem la llista dicc a generar_dataset
		arr = generar_dataset(nombre_ciclistes, index, dicc)
		self.assertEqual(len(arr), nombre_ciclistes)

	def test_valorsmitjatp(self):
		"""
		Test del valor mitjà del temps de pujada
		dintre del rang esperat
		"""
		nombre_ciclistes = 150
		index = 1
		# Passem la llista completa de categories a la funció
		# arr = generar_dataset(nombre_ciclistes, index, dicc[0])
		arr = generar_dataset(nombre_ciclistes, index, dicc)        
		#assertLess(a, b)
		#assertGreater(a, b)
		#arr.append(generar_dataset(100, 1, dicc[3]))        
		# la columna tp és la segona
		# entenc que es el temps de pujada per la
		# segona columna
		# li passem el nom de la columna temps_pujada 
		arr_tp = [row["temps_pujada"] for row in arr] 
		tp_mig = sum(arr_tp)/len(arr_tp)
		# self.assertLess(tp_mig, 3400)
		self.assertLess(tp_mig, dicc[0]["mu_p"] + 3 * sigma)
		self.assertGreater(tp_mig, dicc[0]["mu_p"] - 3 * sigma)

	def test_valorsmitjatb(self):
		"""
		Test del valor mitjà del temps de baixada
		dintre el rang esperat
		"""
		nombre_ciclistes = 150
		index = 1
		# Passem la llista completa de categories a la funció
		arr = generar_dataset(nombre_ciclistes, index, dicc)
		# li passem el nom de la columna temps_baixada
		arr_tb = [row["temps_baixada"] for row in arr]
		tb_mig = sum(arr_tb)/len(arr_tb)
		# self.assertGreater(tb_mig, 2000)
		self.assertLess(tb_mig, dicc[1]["mu_b"] + 2 * sigma)
		self.assertGreater(tb_mig, dicc[1]["mu_b"] - 2 * sigma)        

class TestClustersCiclistes(unittest.TestCase):
	"""
	classe TestClustersCiclistes
	"""
	global ciclistes_data_clean
	global data_labels

	path_dataset = './data/ciclistes.csv'
	ciclistes_data = load_dataset(path_dataset)
	ciclistes_data_clean = clean(ciclistes_data)
	true_labels = extract_true_labels(ciclistes_data_clean)
	# eliminem el tipus, ja no interessa
	ciclistes_data_clean = ciclistes_data_clean.drop('dorsal', axis=1) 
	ciclistes_data_clean = ciclistes_data_clean.select_dtypes(include=['float64', 'int64'])

	clustering_model = clustering_kmeans(ciclistes_data_clean)
	with open('model/clustering_model.pkl', 'wb') as f:
		pickle.dump(clustering_model, f)
	data_labels = clustering_model.labels_


	def test_check_column(self):
		"""
		Comprovem que la columna tp (temps de pujada existeix)
		"""

		self.assertIn('temps_pujada', ciclistes_data_clean.columns)

	def test_data_labels(self):
		"""
		Comprovem que data_labels té la mateixa longitud que ciclistes
		"""

		self.assertEqual(len(data_labels), len(ciclistes_data_clean))

	def test_model_saved(self):
		"""
		Comprovem que a la carpeta model/ hi ha els fitxer clustering_model.pkl
		"""
		check_file = os.path.isfile('./model/clustering_model.pkl')
		self.assertTrue(check_file)

if __name__ == '__main__':
	unittest.main()
