
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
from termcolor import colored

app = Flask(__name__)

def generatLog(message, type):
    f=open("logs/logsData.log", "a")
    logStr=message + " - " + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + ";" + "\n"
    f.write(logStr)
    if(type==1):
        print(colored(logStr, 'yellow'))
    elif(type==2):
        print(colored(logStr, 'red'))
    f.close()

@app.route("/predictOne", methods=['POST'])
def predictOne():

    #Se carga el Pipeline
    fe_pipeline = joblib.load("pkl/titanic_pipeline_v27112022.pkl")
    model_fe_pipeline = joblib.load("pkl/model_titanic_pipeline_v27112022.pkl")
    generatLog("INFO - FE cargado exitosamente", 1)
    #Se lee el csv
    dataTrain = pd.read_csv('dataset/train.csv')
    generatLog("INFO - Dataset cargado exitosamente", 1)

    #Se convierte a tipo objecto algunas variables
    dataTrain['Pclass'] = dataTrain['Pclass'].astype('O')
    dataTrain['Sex'] = dataTrain['Sex'].astype('O')
    dataTrain['SibSp'] = dataTrain['SibSp'].astype('O')
    dataTrain['Parch'] = dataTrain['Parch'].astype('O')
    dataTrain['Embarked'] = dataTrain['Embarked'].astype('O')
    dataTrain['Survived'] = dataTrain['Survived'].astype('O')
    generatLog("INFO - Variables transformadas a objetos exitosamente.", 1)

    #Se hace split de la data
    X_train, X_test, y_train, y_test = train_test_split(dataTrain.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1,), dataTrain['Survived'], test_size=0.3, random_state=2022)
    generatLog("INFO - Split de la data", 1)
    
    #Se usa el Pipeline de FE para transformar los datasets
    X_train_transform = fe_pipeline.fit_transform(X_train)
    X_test_transform = fe_pipeline.fit_transform(X_test)  
    generatLog("INFO - Modulo de FE utilizado exitosamente", 1) 
    
    #Modelo de regresión logistica
    lr=LogisticRegression(max_iter=1000)
    lr.fit(X_train_transform, y_train.astype('int'))
    lr_preds = lr.predict(X_test_transform)
    lr_cm = accuracy_score(y_test.astype('int'), lr_preds)
    generatLog("INFO - Modelo Logistic regression", 1) 

    #KNeighborsClassifier
    knn = KNeighborsClassifier(algorithm = 'brute', n_jobs=-1)
    knn.fit(X_train_transform, y_train.astype('int'))
    knn_preds = knn.predict(X_test_transform)
    knn_cm = accuracy_score(y_test.astype('int'), knn_preds)
    generatLog("INFO - Modelo KNeighborsClassifier", 1) 

    #SVM
    svm = LinearSVC(C=0.0001)
    svm.fit(X_train_transform, y_train.astype('int'))
    svm_preds = svm.predict(X_test_transform)
    svm_cm = accuracy_score(y_test.astype('int'), svm_preds)
    generatLog("INFO - Modelo SVM", 1) 

    #Decision Tree
    clf = DecisionTreeClassifier()
    clf.fit(X_train_transform, y_train.astype('int'))
    clf_preds = clf.predict(X_test_transform)
    clf_cm = accuracy_score(y_test.astype('int'), clf_preds)
    generatLog("INFO - Modelo Decision Tree", 1) 

    #Random Forest
    rf = RandomForestClassifier(n_estimators=30, max_depth=9)
    rf.fit(X_train_transform, y_train.astype('int'))
    rf_preds = rf.predict(X_test_transform)
    rf_cm = accuracy_score(y_test.astype('int'), rf_preds)
    generatLog("INFO - Modelo Random Forest", 1) 

    #Se trabaja conlos parametros
    data=request.get_json()
    lr_parameters=''
    if(data):
        dataFrame = pd.json_normalize(data)
        ids=dataFrame['PassengerId'].astype('int')
        FEATURES = joblib.load("pkl/FEATURES.pkl")
        dataFrame = dataFrame[FEATURES]
        dataFrame['Pclass'] = dataFrame['Pclass'].astype('O')
        dataFrame['Sex'] = dataFrame['Sex'].astype('O')
        dataFrame['SibSp'] = dataFrame['SibSp'].astype('O')
        dataFrame['Parch'] = dataFrame['Parch'].astype('O')
        dataFrame['Embarked'] = dataFrame['Embarked'].astype('O')
        #dataFrame['Age'] = dataFrame['Age'].astype('float')
        generatLog("INFO - Parametros obtenidos y transformados exitosamente.", 1)
        try:
            preds = model_fe_pipeline.predict(dataFrame)    
            out={}
            for index, item in enumerate(preds):
                out[ids[index]]=item    
            return jsonify({
                'Regresion logistica': lr_cm,
                'K-Nearest Neighbours': knn_cm,
                'SVM': svm_cm,
                'Decision Tree': clf_cm,
                'Random Forest': rf_cm,
                'Prediccion': str(out)
            })
        except ValueError:
            #print(NameError)
            generatLog("ERROR - Se ha producido un error", 2)
            return jsonify({'Mensaje': 'ERROR - Se ha producido un error'})
    else:
        return jsonify({
            'Regresion logistica': lr_cm,
            'K-Nearest Neighbours': knn_cm,
            'SVM': svm_cm,
            'Decision Tree': clf_cm,
            'Random Forest': rf_cm
        })
    