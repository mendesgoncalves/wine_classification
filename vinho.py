#carregando o conjunto de dados
import pandas as pd
arquivo = pd.read_csv('/home/mendes/Desktop/wine/wine_dataset.csv')
arquivo['style'] = arquivo['style'].replace('red',0)
arquivo['style'] = arquivo['style'].replace('white',1)
y = arquivo['style']
x = arquivo.drop('style' , axis = 1)

from sklearn.model_selection  import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size= 0.3) #pegando 30% dos dados  para o treino

from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(x_treino,y_treino)
resultado = modelo.score(x_teste,y_teste)

#mostrando a precisao de acerto
print('Acurácia: ', resultado)

#pegando 3 amostras
previsao = modelo.predict(x_teste[400:403])
print(previsao)