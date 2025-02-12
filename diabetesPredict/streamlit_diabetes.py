#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
import matplotlib as plt

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format

# 1. 데이터 로드 및 전처리
data = pd.read_csv('dataset/diabetes.csv')

# 선택된 feature만 사용
selected_features = ['Glucose', 'BMI', 'Age']
X = data[selected_features]
y = data['Outcome']  # 예측할 대상

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 랜덤포레스트 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 모델 저장
joblib.dump(model, 'diabetes_model.pkl')

# 테스트 데이터로 정확도 확인
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# 3. Streamlit 앱
st.title('당뇨병 예측 시스템')
st.write('Glucose, BMI, Age 값을 입력하여 당뇨병 예측을 해보세요.')

# 사용자 입력받기
glucose = st.slider('Glucose (혈당 수치)', min_value=0, max_value=200, value=100)
bmi = st.slider('BMI (체질량지수)', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
age = st.slider('Age (나이)', min_value=0, max_value=100, value=30)

# 예측하기 버튼
if st.button('예측하기'):
    # 입력값을 모델에 전달
    model = joblib.load('diabetes_model.pkl')
    input_data = np.array([[glucose, bmi, age]])
    prediction = model.predict(input_data)[0]

    # 결과 출력
    if prediction == 1:
        st.write('예측 결과: 당뇨병 가능성이 높습니다.')
    else:
        st.write('예측 결과: 당뇨병 가능성이 낮습니다.')

#streamlit run streamlit_diabetes.py


