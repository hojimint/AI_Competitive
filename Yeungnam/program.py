import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import model_selection
from sklearn  import decomposition
from sklearn.preprocessing import LabelEncoder
def data_set():
    
    df = pd.read_csv("train_8_final.csv")
    print(df.info())
    print(df["class"].unique())
    df1 = df["class"]
    df = df.drop(columns=["class"])

    df_filled = df.fillna(df.mean())
    # 결측값 개수 확인
    
    # print(df.isnull().sum())
    # 결측값 평균 계산
    
    v = df.mean(numeric_only=True)
    # # 결측값 채우기
    
    df.fillna(v, inplace = True)
    # # 다시 결측값 개수 확인
    df = pd.concat([df, df1], axis=1)
    # print(df.info())

    return df
data_set()


def get_data_train():

    # 위에서 가공한 df 받아오기
	
    df = data_set()
    # print(df.info())
    
    # 데이터프레임을 numpy 배열로 변경

    data = df.values
    x = data[:,:-1]
    y= data[:,-1]    
    # print(x.shape)
    # print(y.shape)    
    
    #타겟 데이터(y)를 인코딩하기 위한 객체를 생성
    label_encoder = LabelEncoder()
    y_en = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, 'encoder.tool')
    # print(label_encoder.classes_)

	# x 데이터 차원 축소
    # decom = decomposition.PCA(n_components=2)
    # x = decom.fit_transform(x)
    # joblib.dump(decom, 'decom.tool')

	# 데이터 스케일링
    sc = preprocessing.StandardScaler() 
    # sc = preprocessing.Normalizer()
    # sc = preprocessing.MinMaxScaler()       
    x = sc.fit_transform(x)
    joblib.dump(sc, 'scale.tool')
    
    # 학습 데이터와 테스트 데이터를 분할
    x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y_en,test_size=0.2,random_state=2023)  


    return x_train,x_test,y_train,y_test

# get_data_train()

from sklearn import tree, svm, neighbors, naive_bayes, ensemble
import xgboost as xgb
def get_model():
   
    # model = tree.DecisionTreeClassifier(random_state=2023)
    # model = tree.ExtraTreeClassifier(random_state=2023)
    # model = svm.SVC(random_state=2023) 
    # model = neighbors.KNeighborsClassifier()
    # model = naive_bayes.
    model = ensemble.RandomForestClassifier(max_depth = 20, min_samples_split = 5, n_estimators = 200,random_state = 2023)
    # model = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 3, n_estimators = 300,use_label_encoder=False,eval_metric='error',random_state = 2023)
    return model
# get_model()

def train_model():
    x_train,x_test,y_train,y_test = get_data_train()
 #  -------------SVC()-------------------   

    # params={
    #     'C':[0.1,1.0,10],
    #     'kernel':['linear','poly','rbf'],
    #     'gamma':['scale','auto']
    # }
    # models = model_selection.GridSearchCV(estimator=svm.SVC(),param_grid=params,cv=5)
#  -------------DecisionTreeClassifier()-------------------
    # params = {
    # 'max_depth': [3, 5, 7, 10], 
    # 'min_samples_split': [2, 5, 10]
    # }
    # models = model_selection.GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=params, cv=5)
    # models.fit(x_train,y_train)
#  -------------ExtraTreeClassifier()-------------------
    # params = {
    #     'max_depth': [3, 5, 7, 10],  # 여러 후보값들을 넣어보세요
    #     'min_samples_split': [2, 5, 10],  # 여러 후보값들을 넣어보세요
    #     'min_samples_leaf': [1, 2, 4]  # 여러 후보값들을 넣어보세요
    # }
    # models = GridSearchCV(estimator=tree.ExtraTreeClassifier(), param_grid=params, cv=5)
    # models.fit(x_train,y_train)
#  -------------KNeighborsClassifier()-------------------
    # params = {
    #     'n_neighbors': [3, 5, 7, 10],  # 이웃의 수에 대한 여러 후보값을 넣어보세요
    #     'weights': ['uniform', 'distance']  # 가중치에 대한 여러 후보값을 넣어보세요
    # }
    # models = GridSearchCV(estimator=neighbors.KNeighborsClassifier(), param_grid=params, cv=5)
    # models.fit(x_train,y_train)
#  -------------xgb.XGBClassifier()-------------------
    model = xgb.XGBClassifier(use_label_encoder=False,eval_metric='error',random_state = 2023)
    params = {
        'n_estimators': [100, 150, 200, 250, 300],  # 나무의 수에 대한 여러 후보값들을 넣어보세요
        'max_depth': [2,3,4],  # 나무의 최대 깊이에 대한 여러 후보값들을 넣어보세요
        'learning_rate': [0.2 ,0.1, 0.05 ]  # 학습률에 대한 여러 후보값들을 넣어보세요
    }
    models = GridSearchCV(estimator=model, param_grid=params, cv=5)
    models.fit(x_train,y_train)
    print('best score:',models.best_score_ )
    print('best param:',models.best_params_ )

train_model()

def do_train():
    x_train,x_test,y_train,y_test = get_data_train()

    model = get_model()

    model.fit(x_train,y_train)

    score = model.score(x_test,y_test)
    print('score:',score)
    joblib.dump(model,'t4.model')

    return model
# do_train()


def get_data_test():
    df = pd.read_csv('test_2.csv')
    # print(df.info())


    data = df.values
    x = data[:,:]
    model1 = joblib.load('scale.tool')
    # model2 = joblib.load('decom.tool')
    x = model1.fit_transform(x)
    # x = model2.fit_transform(x)

    return x
# get_data_test()

def do_predict():

    x = get_data_test()
    # print(x.shape)
    model1 = joblib.load('t4.model')
    y_pre = model1.predict(x)

    # model2 = joblib.load('encoder.tool')
    # original_values = model2.inverse_transform(y_pre)
    print(y_pre,y_pre.shape)
    y_pre = y_pre.reshape(-1,1)
    print(y_pre.shape)


    df = pd.DataFrame(y_pre,columns=['class'])
    # 라벨 인코딩된 값을 기본의 문자로 변환
    df['class'] = df['class'].replace({0: 'B', 1: 'M'})


    df.to_csv('홍길동.csv',index=False)

do_predict()
