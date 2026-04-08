import pandas as pd
from sklearn.datasets import load_breast_cancer
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression



dt = load_breast_cancer()
x = pd.DataFrame(dt.data,columns=dt.feature_names)
y = dt.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)

pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('feature_selection',RFE(estimator=LogisticRegression(max_iter=500),n_features_to_select=10)),
    ('model',LogisticRegression(max_iter=500))
])

pipeline.fit(x_train,y_train)

joblib.dump(pipeline,'cancer_pipeline.pkl')
print('model saved successfully')




