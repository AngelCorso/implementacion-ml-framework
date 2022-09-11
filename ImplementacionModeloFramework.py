# Librerias
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cargar datos
iris = datasets.load_iris()
list(iris.keys())

# Creación y separación de datos
X = pd.DataFrame(iris['data'], columns = iris['feature_names'])
y = pd.DataFrame(iris['target'], columns = ['species'])

# Separación del conjunto de datos para prueba y valicación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_train, y_val = y_train['species'], y_test['species']

# Creación del modelo
clf = RandomForestClassifier(n_estimators = 100, random_state = 42)

# Entrenamiento de datos
clf.fit(X_train, y_train)

# Predicción
Y_hat = clf.predict(X_test)

# Accuracy del modelo
print('Accuracy: ', accuracy_score(y_test, Y_hat))