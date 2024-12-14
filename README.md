# Table of contents
1. [portcanto](#portcanto)
2. [Docker](#docker)
3. [Generar_conjunt_de_dades](#Generar_conjunt_de_dades)
4. [Clustering](#clustering)
5. [Guardar_model](#pickle)
6. [Generacio_dels_informes](#informes)
7. [Analisi_de_codi_estatic](#pylint)
8. [Documentacio](#docstrings)
9. [Testing](#tests)
10. [Prediccio_de_nous_valors](#prediccio)
11. [MLflow](#MLflow)
12. [Llicencia](#licence)


# portcanto <a name="portcanto"></a>

El Port del Cantó és un port de muntanya que uneix les comarques de l’Alt Urgell (Adrall) i el Pallars Sobirà (Sort). 
Són 18Km de pujada i 18Km de baixada, que típicament es puja entre 54 i 77min; i es baixa entre 24 i 36min. Es 
generaran dades sintètiques que simularan una cursa ciclista entre Adrall i Sort. Es treballarà sobre aquestes dades.

**portcanto** és un projecte de simulació de ciclistes en funció de les seves carácteristiques per ser escaladors o baixadors. 
S'han definit 4 patrons de ciclistes (BEBB, BEMB, MEBB i MEMB).
L'objectiu és descobrir els 4 patrons amb l'algoritme de clustering KMeans.

ENG: This is a Python, it is a project to simulate cyclists depending on their characteristics 
to be climbers or descenders. 
4 patterns of cyclists have been defined (BEBB, BEMB, MEBB and MEMB).
The goal is to discover the 4 patterns with the KMeans clustering algorithm.

Es vol crear dades sintètiques per poder fer un anàlisi de les dades amb IA (bàsicament un problema de clustering).

Consta de dues parts:

- generardataset: crea a la carpeta data/ dades simulades. En principi es simulen 4 comportaments de ciclistes, 
  que donaran lloc a 4 categories/clústers.
- clustersciclistes: crea els diferents clústers en funció dels diferents tipus de ciclistes:
- BEBB: Bons escaladors, bons baixadors.
- BEMB: Bons escaldors, mals baixadors.
- MEBB: Mals escaladors, bons baixadors.
- MEMB: Mals escaladors, mals baixadors.

Consta també de diferents carpetes i scripts relacionats amb l'anàlisi de les dades i 
la solució amb IA del problema del clustering.

# Docker <a name="docker"></a>

```
Descarrega:
$ docker pull iabdioc/docker-ia:latest

A partir de l'esquelet proporcionat en l'enunciat, el copiarem dintre del directori volum
de la última versió del docker-ia que hem de carregat.

Crea i arrenca el contenidor:
$ docker run -dit -p 8888:8888 -p 5000:5000 --name ARRIASOLEAC6 -v ~/volum:/home/dockeria/volum -v /tmp/.X11-unix:/tmp/.X11-unix iabdioc/docker-ia:latest
  * port 8888: Podem accedir a Jupyter Notebook.
  * port 5000: Podem executar (inicialitzar) el servidor de MLflow.

Accedir al contenidor
$ docker exec -it portcanto /bin/bash

i tot seguit instal·lar els mòduls necessaris:
```
$ pip install -r requirements.txt
```

Dins del contenidor pots arrencar el servidor web de Jupyter:
$ arrencar_jupyter.sh

Pots testejar un quadern de Jupyter a:
volum/portcanto/generardataset.py

També pots arrancar el servidor MLflow per comprovar si la interfice
gràfica funciona correctament amb la ordre:
$ mlflow ui --host = 0.0.0.0
O amb l'ordre 
$ mlflow server --host = 0.0.0.0
```

# Generar_conjunt_de_dades <a name="Generar_conjunt_de_dades"></a>

Generem un conjunt de dades per a ciclistes en base a les categories.
Creem un fitxer CSV amb dades sobre els temps de pujada i baixada
de ciclistes, basat en un diccionari amb mitjanes i desviacions estàndard
per a diverses categories de ciclistes.
Des de l'arrel del projecte hem executat:
```
$ python generardataset.py
```

# Clustering <a name="clustering"></a>

Analitzem el comportament de ciclistes utilitzant clustering KMeans.
Aquest script proporciona funcions per carregar dades, netejar-les, aplicar clustering,
i generar informes amb visualitzacions gràfiques.

Dins les carpetes <em>portcanto</em> i es resol el problema de trobar els clústers per a les dades simulades.
Fem servir al script clustersciclistes.py

Des de l'arrel del projecte hem executat:
```
$ python clustersciclistes.py
```

S'obtenen 4 clústers.

![Clusters](clusters.png)
![Pairplot](pairplot.png)

# Guardar_model <a name="pickle"></a>

La biblioteca Pickle serveix per exportar models a un fitxer. Convertim l'objecte a bytes, 
aquest procés s'anomena serialitzar. 
D'aquesta manera el conjunt de bytes es pot guardar fàcilment en un fitxer. 
I podem reutilitzar-los sense la necessitat 
de tornar-los a entrenar cada cop que els necessitem.

Dins la carpeta <em>model</em> trobarem els següents models:

- clustering_model.pkl
- scores.pkl
- tipus_dict.pkl

# Generacio_dels_informes <a name="informes"></a>

Hem generat 5 fitxers. Un fitxer amb les dades generals anomenat “tots_els_clusters.txt”  i 
4 fitxers més un per cada clúster.
En el fitxer “tots_els_clusters.txt”  disposem d’un recull dels resultats obtinguts del nostre 
model de clustering. On podem veure els clústers que hem obtingut del nostre model. En el 
nostre cas 4 perquè li hem indicat que volíem dividir les dades en 4 clústers o grups, 
amb les seves etiquetes respectives. Alhora també podem veure a quina etiqueta correspon 
cada tipus (BEBB=etiqueta 3, BEMB=etiqueta 0, MEBB= etiqueta 2  i MEMB= etiqueta 1).
En aquest fitxer també disposem de les característiques de cada clúster com son la mitjana 
de les característiques  “temps_pujada” i “temps_baixada”.
En cadascun dels altres 4 fitxers hi ha informació de les característiques i els 
punts de dades associats.
En el clúster 0 hi ha les dades associades al tipus BEMB(patró), al clúster 1 hi ha les dades 
associades al tipus MEMB (patró), al clúster 2 hi ha les dades associades al tipus MEBB(patró) i 
al clúster 3 hi ha les dades associades al tipus BEBB(patró).
En aquest 4 fitxers hi ha una descripció del tipus de comportament associat i les dades 
concretes de cada clúster.
Tots aquests fitxers en cada observació, fila, hi ha informació sobre el temps de pujada, 
el temps de baixada i l’identificador del clúster (etiqueta).
Finalment, comentar que en aquests fitxers podem veure les tendències en funció de cada clúster 
i les dades concretes de cada usuari que pertany al clúster.
En el nostre cas teníem 150 usuaris o ciclistes que s’han dividit en funció de les seves dades en 
el clúster, agrupació, corresponent.

Dins la carpeta <em>informes</em> trobarem els següents informes:

- tots_els_clusters.txt
- cluster_0.txt
- cluster_1.txt
- cluster_2.txt
- cluster_3.txt


# Analisi_de_codi_estatic <a name="pylint"></a>

Fem servir eines de linting (anàlisi estàtica).
Amb les qual complim amb les bones pràctiques de codificació i d’estils en el nostre 
cas amb Python.
Examinen el codi per detectar errors de sintaxi, codi incorrecte, males pràctiques i així 
seguir les normes d’estil. Com a errors habituals citar longitud de la línia excessiva, 
identació incorrecta, espais en blanc addicionals, falta d’espais en blanc al voltant dels operador, 
noms de variables no descriptius, importacions incorrectes, comentarios inapropiats, ignorar les 
convencions de com hem d’anomenar les variables i funcions, no seguir l’estil dels docstrings.


Des de l'arrel del projecte hem executat:
```
$ pylint generardataset.py
$ pylint clustersciclistes.py
```

# Documentacio <a name="docstrings"></a>

Per generar la documentació des de l'arrel del projecte executem:
```
$ python -m pydoc -w generardataset.py
$ python -m pydoc -w clustersciclistes.py
```

# Testing <a name="tests"></a>

Els tests unitaris (els farem amb la llibreria estàndard de python anomenada unittest) ens permeten fer casos 
de prova, configuració i neteja. És a dir ens permeten comprovar les funcions del nostre codi, trobar errors 
i facilitar el desenvolupament del nostre programa. Podem comprovar els casos que creguem necessaris repassar 
per veure si son correctes.
D’aquesta manera podrem anar resolent les errades mentrestant es van desenvolupant i  així no es necessari torna 
a començar. Resumint es tracta de codi que testeja quan un altre codi es comporta correctament.
En el nostre cas hem pogut fer proves sobre els scripts generardataset i clustersciclistes. Els nostres tests 
s’executaran dintre del directori portcanto. Amb els asserts comprovem si les proves realitzades s’ha adeqüen
amb les dades dels scripts que volem comprovar.

Per realitzar els tests des de l'arrel del projecte executem:
```
$ python -m unittest discover -s tests
```

# Prediccio_de_nous_valors <a name="prediccio"></a>

Comentem que els fitxers que es guarden en el directori model son els que farem servir 
per la nostra predicció.
-	clustering_model.pkl : És el model d’entrenament de clustering que hem fet 
    servir amb sklearn kmeans.
-	scaler.pkl : Abans de realitzar l’entrenament hem fet la normalització de les dades. 
    Per tant es important guardar aquest model perquè quan volguem fer una nova predicció 
	les dades s’hauran de normalitzar amb el mateix model.
-	tipus_dict.pkl : Guardem el diccionari tipus que ens tradueix els labels (0,1,2,3,4) al 
    tipus de ciclista (BEBB, BEMB, MEBB i MEMB).
	
Per realitzar la predicció des de l'arrel del projecte executem:	
```
$ python prediccio.py
```

# MLflow <a name="MLflow"></a>

Farem un experiment, en el qual provarem de fer el procés de clustering
amb un valor de k entre 2 i 8.
A part de veure el resultat per pantalla també el veure'm de forma gràfica.

Primer executarem la següent ordre per arrancar el entorn gràfic, on 
posteriorment podrem comprovar els resultats des del navegador web
des de l'adreca : http://localhost:5000

Per arrancar l'entorn gràfic executarem la següent ordre:
```
$ mlflow ui --host = 0.0.0.0
```

Fet lo anterior executem el script clustersciclistes:
```
$ python clustersciclistes.py
```

Finalment executem al script mlflowtracking-K.py es on hem
preparat el nostre experiment.
```
$ python mlflowtracking-K.py
```

Un cop ja hem executat el nostre experiment ja podem carregar
l'entorn gràfic al nostre navegador (http://localhost:5000)
I comprovar els diferents resultats pels diferents valor de K.
És a dir, comprovar els diferents valors en funció dels clústers
(entre 2 i 8).

# Llicencia <a name="licence"></a>

Àngel Arriasol (2024)
Llicència MIT. [LICENSE.txt](LICENSE.txt) per més detalls.


