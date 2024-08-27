import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt

def dividir_datos_y_target(X, y, tamaño_prueba=0.2, random_state=None):
 
    # Convertir a numpy arrays si son listas
    X = np.array(X)
    y = np.array(y)
    
    # Fijar la semilla aleatoria para reproducibilidad
    if random_state is not None:
        np.random.seed(random_state)
    
    # Mezclar los índices de los datos
    indices = np.arange(len(X)) # Aqui genero los indices que se mezclarán
    np.random.shuffle(indices) #Aqui se mezclan
    
    # Mezclar X e y en función de los índices
    X = X[indices] 
    y = y[indices]
    
    # Calcular el índice de corte
    test_size_index = int(len(X) * tamaño_prueba) # Este numero debe de ser entero y es el que se encarga de dividir el tamaño del
                                                  # dataset de prueba y de entrenamiento
    
    # Dividir los datos y etiquetas en entrenamiento y prueba
    X_test = X[:test_size_index]
    X_train = X[test_size_index:]
    y_test = y[:test_size_index]
    y_train = y[test_size_index:]
    
    return X_train, X_test, y_train, y_test # Me devuelve 4 variables, los features de entrenamiento y prueba, y los labels de entrenamiento y prueba 

class EscaladorDatos: # Lo hice en una clase para que sea mas sencillo llamar a la función y se utiliza para mejorar el performance del 
                      # modelo y acelerar la convergencia
    def __init__(self):
        self.media = None 
        self.stds = None

    def fit(self, X):
        #Calcula la media y la desviación estándar de cada feature en los datos X.

        X = np.array(X) # Se transforman los datos de X a un array en caso de que no lo sea
        
        # Calcular la media y desviación estándar de cada columna (feature)
        self.media = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)

    def transform(self, X):
        # Transforma los datos X restando la media y dividiendo por la desviación estándar calculada.
        
        # Asegurarse de que fit haya sido llamado antes
        if self.media is None or self.stds is None:
            raise Exception("El método 'fit' debe ser llamado antes de 'transform'.")
        
        X = np.array(X)
        
        # Normalizar los datos: (X - media) / desviación estándar
        X_escalada = (X - self.media) / self.stds
        
        return X_escalada

    def fit_transform(self, X):
        #Ajusta y transforma los datos X.
        
        self.fit(X)
        return self.transform(X)

def sigmoid_function(X): #Esta es la función sigmoide, recibe como parametros una combinación lineal de elementos para posteriormente
                         # transformarla en pesos que representan probabilidades.
  return 1/(1+math.e**(-X)) 

def log_regression5(X, y, alpha, epochs): # Función que realiza la función logistica donde x son los samples de nuestros features,
                                          # y nuestros labels, alpha la tasa de aprendizaje, y epochs la cantidad de iteraciones
                                          # maximas en las que se realizará el entrenamiento del modelo.
    
  y_ = np.reshape(y, (len(y), 1)) # Se cambia la forma del target (y) de tal manera que quede en forma de columna.
  N = len(X)                      # Se calcula la cantidad de damples de los features para poder realizar el metodo de optimización 
                                  # gradient descent.
                                  
  theta = np.random.randn(len(X[0]) + 1, 1) # Theta son los pesos de o w de la regresión logistica, no obstante al principio
                                            # esteblecemos datos aleatorios que posteriormente serán optimizados por el gradient
                                            # descent.
                                            
  X_vect = np.c_[np.ones((len(X), 1)), X] # Al vector de los features que contiene los samples, se le agrega el bias.
  
  avg_loss_list = [] # En esta lista se almacenará la perdida promedio de la función de costo.
  
  for epoch in range(epochs): # Se inicia el ciclo for en el que se realizará el entrenamiento del modelo de machine learning 
                              # en el maximo de epochs
                              
    sigmoid_x_theta = sigmoid_function(X_vect.dot(theta)) # Debido a que estamos haciendo una regresión logística, utilizamos la función
                                                          # sigmoide, ya que transforma un resultado lineal en una probabilidad, se 
                                                          # utiliza para clasificación binaria, y es diferenciable y apta para optimización
                                                          # que para este caso utilizaremos gradient descent.
                                                          # De input se utilizó el producto punto de los features con el cesgo con la theta, 
                                                          # que representan los pesos (los cuales representan una probabilidad de que 
                                                          # salga algún valor de la clasificación binaria).
                                                          
    grad = (1/N) * X_vect.T.dot(sigmoid_x_theta - y_) # Aqui utilizamos decenso de gradiente como función de optimización, dividimos  
                                                      # todo entre N debido a que queremos optimizar todo el data frame, multiplicado
                                                      # por el producto punto de x_vect (contiene el cesgo y los samples de nuestros 
                                                      # features), con sigmoid_x_theta - y_ (representan la resta de los theta o pesos
                                                      # que nos dio el utilizar la función sigmoide con los valores de nuestro label),
                                                      # esto representa una parte de la ecuación completa del gradient descent.

    theta = theta - (alpha * grad) # Esto representa la ecuación completa del metodo de optimización por decenso de gradiente.
    
    best_params = theta # Estos son los valores optimos de theta después de utilizar el gradient descent

    hyp = sigmoid_function(X_vect.dot(theta)) # Declaramos la función de hipotesis la cual se utilizará para la función de costo 
    
    avg_loss = -np.sum(np.dot(y_.T, np.log(hyp) + np.dot((1-y_).T, np.log(1-hyp)))) / len(hyp) # Calculamos la perdida promedio de la 
                                                                                               # función de hipotesis utilizando la
                                                                                               # función de perdida logaritmica negativa
                                                                                               # (o negative log loss function en ingles)
    
    if epoch == 0:
      avg_loss_list.append(avg_loss) # Hacemos explicito que en la primera iteración se guarde el valor de perdida promedio
      
    elif avg_loss_list[-1]-avg_loss < 0.001: # En este if lo que hacemos es establecer que si la diferencia de perdida promedio
                                             # entre el ultimo valor registrado de la lista de perdida promedio y el error promedio
                                             # actual es menor de 0.001, que salga del ciclo y termine la operación, con el proposito
                                             # de evitar que el modelo siga intentando optimizar cuando ya se encontró una solución 
                                             # optima.
                                             
      print('epoch: {} | avg_loss: {}'.format(epoch, avg_loss)) # Imprimimos la epoch o iteración en el cual encontramos el parametro
                                                                # optimo y el error promedio encontrado.
      break # Sale del ciclo for
      
    avg_loss_list.append(avg_loss) # Una vez calculada la perdida promedio, se procede a almacenarla en la lista.
    
  # Mostramos la gráfica de comportamiento de la función de costo, el cual debe de mostrar que va disminuyendo hasta encontrar un minimo
  plt.plot(np.arange(0, epoch), avg_loss_list[1:], color='red') 
  plt.title('Función de costo')
  plt.xlabel('Epochs o iteración')
  plt.ylabel('Costo')
  plt.show()
  
  # Lo que regresa la función son los pesos de la función sigmoide, que representan las probabilidades de que salga un valor del target,
  # lo cual resulta util en este caso debido a que el problema es de clasificación binaria.
  return best_params 

df = pd.read_csv('framingham.csv') # Dataset extraido de https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression

df = df.replace(np.nan, df.mean()) # Reemplazamos todos los valores nan del data frame, con la media del feature.

print(df) # Se imprime el data frame 

x = df.drop('TenYearCHD', axis=1) # De features escogemos todos menos el label.

# Regresa 1 si no hay riesgo de fallo cardiaco (0 = no hay riesgo de fallo cardiaco en 10 años, 1 = si hay riesgo de fallo cardiaco 
# en 10 años) si si lo hay.
y_0 = (df["TenYearCHD"] == 0).astype(int) 
y_1 = (df['TenYearCHD'] == 1).astype(int)

# Se divide el data set en entrenamiento y prueba, de tal manera que el 20% de nuestros datos sea de prueba y el 80% de entrenamiento
X_train0, X_test0, y_train0, y_test0 = dividir_datos_y_target(x, y_0, tamaño_prueba=0.2, random_state=2)


Escala0 = EscaladorDatos()
X_train0 = Escala0.fit_transform(X_train0) # Aqui escalamos y transformamos  los datos para mejorar el rendimiento del modelo y acelerar
                                           # su convergencia
X_test0 = Escala0.transform(X_test0) # Aqui solo se transformaron los datos debido a que se utiliza la media y la desviación estandar
                                     # de los datos de entrenamiento para transformar estos datos

epochs = 10000 # Numero de epocas de entrenamiento
alpha = 1 # Se declara la tasa de aprendizaje 
best_params0 = log_regression5(X_train0, y_train0, alpha, epochs) # Se llama la función de log_regression5 para encontrar los pesos de
                                                                  # optimizados de la función logística


X_to_predict0 = np.c_[np.ones((len(X_test0), 1)), X_test0] # Le añadimos el cesgo a los datos de prueba de nuestros features
X_to_predict0 = np.dot(X_to_predict0,best_params0) # Realizamos la combinación lineal que utilizaremos de input para la función sigmoide
                                                   # con el producto punto de los parametros optimizados de la función logística y con
                                                   # X_to_predict_0 que contiene los datos de prueba y el cesgo de los features

y_pred0 = np.around(sigmoid_function(X_to_predict0)) # Se realizan las predicciones con la función sigmoide y se establece un umbral
                                                     # que establece que cada valor mayor o igual que 0.5 sea 1, en caso que sea menor
                                                     # que 0.5 que sea 0.

y_precision0 = y_test0-y_pred0[:,0]  # Aqui se restan los valores de prueba con los predichos, todos los valores que sean iguales
                                     # deberán ser 0.
precision0 = (y_precision0 == 0).sum()/len(y_precision0) # Se cuentan la cantidad de 0s en el array, y se saca el porcentaje de esto
                                                         # lo cual nos daría la precisión del modelo ya que los 0s representan los 
                                                         # datos acertados.

## Se repite el mismo proceso pero ahora para los datos de prueba
X_train_predict0 = np.c_[np.ones((len(X_train0), 1)), X_train0] 
X_train_predict0 = np.dot(X_train_predict0,best_params0)

y_pred_train0 = np.around(sigmoid_function(X_train_predict0))

y_precision_train0 = y_train0-y_pred_train0[:,0]
precision_train0 = (y_precision_train0 == 0).sum()/len(y_precision_train0)

# Se imprimen los resultados de la precisión del modelo para el dataset de prueba y para el dataset de entrenamiento
print('la precision del modelo de entrenamiento para predecir si alguien no presentará un fallo cardiaco en 10 años es del: ' + str(round(precision_train0*100,2))+'%')
print('la precision del modelo de prueba para predecir si alguien no presentará un fallo cardiaco en 10 años es del: ' + str(round(precision0*100,2))+'%')

###
'''
Se repite el mismo proceso que arriba, pero ahora con los datos de las personas que si están en riesgo de fallo cardiaco en 10 años
'''
###
X_train1, X_test1, y_train1, y_test1 = dividir_datos_y_target(x, y_1, tamaño_prueba=0.2, random_state=2)

Escala1 = EscaladorDatos()
X_train1 = Escala1.fit_transform(X_train1)
X_test1 = Escala1.transform(X_test1)

epochs = 10000
alpha = 1
best_params1 = log_regression5(X_train1, y_train1, alpha, epochs)



X_to_predict1 = np.c_[np.ones((len(X_test1), 1)), X_test1] 
X_to_predict1 = np.dot(X_to_predict1,best_params1)

y_pred1 = np.around(sigmoid_function(X_to_predict1))

y_precision1 = y_test1-y_pred1[:,0]
precision1 = (y_precision1 == 0).sum()/len(y_precision1)


X_train_predict1 = np.c_[np.ones((len(X_train1), 1)), X_train1] 
X_train_predict1 = np.dot(X_train_predict1,best_params1)

y_pred_train1 = np.around(sigmoid_function(X_train_predict1))

y_precision_train1 = y_train1-y_pred_train1[:,0]
precision_train1 = (y_precision_train1 == 0).sum()/len(y_precision_train1)
print('la precision del modelo de entrenamiento para predecir si alguien presentará un fallo cardiaco en 10 años es del: ' + str(round(precision_train1*100,2))+'%')
print('la precision del modelo de prueba para predecir si alguien presentará un fallo cardiaco en 10 años es del: ' + str(round(precision1*100,2))+'%')


# Creamos un diccionario donde el key es el string, y nuestra variable numerica es un array que contiene 1 si la afirmación del string
# es verdad y 0 si no.
y_cardiac_type = {'Riesgo de fallo cardiaco en 10 años':y_1,        
                'No hay riesgo de fallo cardiaco en 10 años':y_0}

# Creamos un diccionario donde el key es el string, y la variable numerica es un float donde se van a almacenar las probabilidades de
# que la afirmación sea verdadera.
predicted_probs = {'Riesgo de fallo cardiaco en 10 años':0.0,
                'No hay riesgo de fallo cardiaco en 10 años':0.0}

# Este diccionario es similar que el anterior, solo que se utilizará para conocer el valor verdadero
actual_y = {'Riesgo de fallo cardiaco en 10 años':0,
                'No hay riesgo de fallo cardiaco en 10 años':0}


for key, y_card_type in y_cardiac_type.items(): # Se hace un ciclo for para cada elemento del diccionario

  # Dividimos el dataset de tal manera que el 20% de los sample sean de prueba y el 80% de entrenamiento
  X_train, X_test, y_train, y_test = dividir_datos_y_target(x, y_card_type, tamaño_prueba=0.2, random_state=2)

  Escala = EscaladorDatos()
  X_train = Escala.fit_transform(X_train) # Se escalan y transforman los datos para mejorarel rendimiento del modelo y acelerar la
                                          # convergencia 
  X_test = Escala.transform(X_test) # Solo se transforman los datos ya que ya se cuenta con la media y desviación estandar de los 
                                    # datos de prueba

  epochs = 10000 # Se declara el numero de epochs o iteraciones
  alpha = 1# Se declara la tasa de aprendizaje
  best_params = log_regression5(X_train, y_train, alpha, epochs) # Se encuentran los mejores parametros
 
  index_ = 10 # Se declara un indice que será utilizado para encontrar un sample de cada feature
  X_to_predict = [list(X_test[index_])] # Se utiliza un sample de cada feature 

  X_to_predict = np.c_[np.ones((len(X_to_predict), 1)), X_to_predict] # Se le añade un bias al sample de los features

  pred_probability = sigmoid_function(X_to_predict.dot(best_params)) # Con el sample que se escogió de cada feature, se calculó la 
                                                                     # probabilidad utilizando la función sigmoide
  predicted_probs[key] = pred_probability[0][0] # Almacenamos la probabilidad predicha en una lista (se hace explicita la posición del 
                                                # de la lista de pred_probability porque es un ndarray no obstante el tamaño de este
                                                # ndarray es (1,1), y nosotros necesitamos el formato de este dato en float)
  
  # Imprimimos la probabilidad de la muestra seleccionada
  print('Se calculó que la probabilidad que la muestra sea {}, es del: {}%'.format(key, round(predicted_probs[key]*100,2))) 
  actual_y[key] = y_test[index_] # Seleccionamos la muestra verdadera, con el indice que utilizamos para seleccionar el sample de los
                                 # feature de prueba

max_key = max(predicted_probs, key=predicted_probs.get) # Seleccionamos el evento con mayor probabilidad
print('\n', predicted_probs)
print('\nPredicción del modelo: {}'.format(max_key)) # Imprimimos el evento con mayor probabilidad como predicción del modelo
max_actual_y = max(actual_y, key=actual_y.get) # Seleccionamos el evento real 
print('Valor real: {}'.format(max_actual_y)) # Imprimimos el evento real 
