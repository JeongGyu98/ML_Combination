import numpy as np
import pandas as pd

from pandas.core import algorithms
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV


def FindBest (X, y, scalers = None, encoders = None, models = None):
    
        '''
        Find the best combination of scaler, encoder, fitting model algorithm
        print best score and best combination 

        Prameters 
        -------------------------------------
        X : DataFrame to scaled 

        y : DataFrame to encoding

        scalers: list of sclaer
            None: [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
            if you want to scale other ways, then put the sclaer in list

        encoders: list of encoder
            encode = [OrdinalEncoder(), StandardScaler()]
            if you want to use only one, put a encoder in list

        models: list of encoder
            classifier = ['DecisionTreeClassifier(gini)', 'DecisionTreeClassifier(entropy)', 'Logistic', 'SVC']
            if you want to fitting other ways, then put in list
        '''

        # For Numeric features and Categorical features
        X_cate = X.select_dtypes(include = 'object')
        df_cate_emtpy = X_cate.empty
        X_nume = X.select_dtypes(exclude = 'object')
        df_nume_empty = X_nume.empty

        if scalers == None:
            scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]   
        else : scale = scalers
        
        if encoders == None:
            encode = [OrdinalEncoder(), StandardScaler()]
        else : encode = encoders

        if models == None:
            classifier = ['DecisionTreeClassifier(gini)', 'DecisionTreeClassifier(entropy)', 'Logistic', 'SVC']
        else : classifier = models

        # Grid Search CV parameters for Decision tree (criterion = gini)
        grid_params_DT_gini = {
            'criterion' : ['gini'],
            'min_samples_split' : [2],
            'max_features' : [3],
            'max_depth' : [3],
            'max_leaf_nodes' : list(range(7, 100))
        }

        # Grid Search CV parameters for Decision tree (criterion = entropy)
        grid_params_DT_entropy = {
            'criterion' : ['entropy'],
            'min_samples_split' : [2],
            'max_features' : [3],
            'max_depth': [3],
            'max_leaf_nodes' : list(range(7, 100))
        }

        # Grid Search CV parameters for Logistic Regression
        grid_params_Logistic = {
            'C' : [0.1, 1, 10],
            'penalty': ['l2']
        }

        # Grid Search CV parameters for SVC
        grid_params_SVC = {
            'C' : [0.001, 0.01, 0.1],
            'gamma' : [0.001, 0.01, 0.1, 1]
        }

        best_score_DT_gini = 0
        best_DT_gini_params = []
        best_DT_gini_scaler = []
        
        best_score_DT_entropy = 0
        best_DT_entropy_params = []
        best_DT_entropy_scaler = []

        best_score_Logistic = 0
        best_Logistic_param = []
        best_Logistic_scaler = []

        best_score_SVC = 0
        best_SVC_param = []
        best_SVC_scaler = []


        # Find best combination using triple loop statements  
        for i in scale :
            for j in encode :

                # If none of data is numeric data, do not scale
                if df_nume_empty is False:
                    scaler = i
                    scaler = pd.DataFrame(scaler.fit_transform(X_nume))

                # If none of data is Categorical data, do not encode
                if j == OrdinalEncoder() and df_cate_emtpy is False :
                    enc = j 
                    enc = enc.fit_transform(X_cate)

                scaler = i
                scaler = pd.DataFrame(scaler.fit_transform(X))

                for model in classifier :
                    
                    X_train, X_test, y_train, y_test = train_test_split(scaler, y, test_size=0.2, random_state=42)

                    if model == 'DecisionTreeClassifier(gini)' :

                        grid_dt_gini = GridSearchCV(DecisionTreeClassifier(), param_grid = grid_params_DT_gini, cv = 3)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                        grid_dt_gini.fit(X_train,y_train)
                        score = grid_dt_gini.score(X_test, y_test)

                        if  best_score_DT_gini < score :
                            best_score_DT_gini = score
                            best_DT_gini_params = grid_dt_gini.best_params_
                            best_DT_gini_scaler = [i, j]
                    
                    elif model == 'DecisionTreeClassifier(entropy)' :

                        grid_dt_entropy = GridSearchCV(DecisionTreeClassifier(), param_grid = grid_params_DT_entropy, cv = 3)
                        grid_dt_entropy.fit(X_train, y_train)
                        score = grid_dt_entropy.score(X_test, y_test)

                        if  best_score_DT_entropy < score :
                            best_score_DT_entropy = score
                            best_DT_entropy_params = grid_dt_entropy.best_params_
                            best_DT_entropy_scaler = [i, j]
                    
                    elif model == 'Logistic':

                        grid_logistic = GridSearchCV(LogisticRegression(), grid_params_Logistic, cv = 3)
                        grid_logistic.fit(X_train, y_train)
                        score = grid_logistic.score(X_test, y_test)

                        if  best_score_Logistic < score :
                            best_score_Logistic = score 
                            best_Logistic_param = grid_logistic.best_params_
                            best_Logistic_scaler = [i, j]

                    if model == 'SVC' : 

                        grid_SVC = GridSearchCV(SVC(), grid_params_SVC, cv = 3)
                        grid_SVC.fit(X_train, y_train)
                        score = grid_SVC.score(X_test, y_test)

                        if  best_score_SVC < score :
                            best_score_SVC = score
                            best_SVC_param = grid_SVC.best_params_
                            best_SVC_scaler = [i, j]
                        
        # Print best scores, parameters, scaler and encoder

        print('Best score for DecisionTree (Gini) :', best_score_DT_gini)
        print('Best parameters' , best_DT_gini_params)
        print('Scaler and Encoder:', best_DT_gini_scaler)

        print('Best score for DecisionTree (Entropy) :', best_score_DT_entropy)
        print('Best parameters' , best_DT_entropy_params)
        print('Scaler and Encoder:', best_DT_entropy_scaler)

        print('Best score for Logistic Regression :', best_score_Logistic)
        print('Best parameters' , best_Logistic_param)
        print('Scaler and Encoder:', best_Logistic_scaler)

        print('Best score for SVC :', best_score_SVC)
        print('Best parameters' , best_SVC_param)
        print('Scaler and Encoder:', best_SVC_scaler)

        return

# Read dataset
df = pd.read_csv('/Users/kim-jeonggyu/Python/Dataset/breastCancer.csv')

# Cleaning dirty data
df_temp = df[df['bare_nucleoli'] == '?'].index
df = df.drop(df_temp)

# Drop column which is not used
df = df.drop(['id'], axis=1)
df = df.reset_index(drop=True)

# Split target data
Y = df['class']
X = df.iloc[:, 0:9]

FindBest(X, Y)

    


