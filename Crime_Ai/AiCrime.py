import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder



# 폰트 지정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
pd.options.display.float_format = '{:.2f}'.format

# Streamlit 페이지 설정
st.set_page_config(page_title="범죄 유형 예측", page_icon="🔍", layout="wide")
st.title("범죄 유형 예측 웹 애플리케이션")

# 1. 데이터 로드 및 전처리
chunksize = 10000  # 한 번에 로드할 데이터 청크 크기
chunks = pd.read_csv('dataset/crime.csv', chunksize=chunksize)
data = pd.concat(chunk for chunk in chunks)  # 모든 청크를 하나의 데이터프레임으로 결합

# 데이터 열 확인
print(data.columns)

# 결측치 통계
st.subheader("결측치 통계")
missing_values = data.isnull().sum()
st.dataframe(missing_values)

# 중복값 통계
st.subheader("중복값 통계")
duplicate_values = data.duplicated().sum()
st.write(f"중복된 행 수: {duplicate_values}")

# 주요 변수별 데이터 분포도
st.subheader("주요 변수별 데이터 분포도")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
sns.histplot(data['YEAR'], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('연도 분포')

sns.histplot(data['MONTH'], bins=12, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('월 분포')

sns.histplot(data['DAY'], bins=31, kde=True, ax=axes[0, 2])
axes[0, 2].set_title('일 분포')

sns.histplot(data['HOUR'], bins=24, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('시간 분포')

sns.histplot(data['MINUTE'], bins=60, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('분 분포')

sns.histplot(data['NEIGHBOURHOOD'], bins=50, kde=True, ax=axes[1, 2])
axes[1, 2].set_title('이웃 분포')

plt.tight_layout()
st.pyplot(fig)

# 필요 없는 열 제거
data = data.drop(['X', 'Y', 'Latitude', 'Longitude'], axis=1)

# 결측치 처리
data = data.dropna()

# 범주형 변수 인코딩
label_encoder = LabelEncoder()
data['TYPE'] = label_encoder.fit_transform(data['TYPE'])
data['HUNDRED_BLOCK'] = label_encoder.fit_transform(data['HUNDRED_BLOCK'])
data['NEIGHBOURHOOD'] = label_encoder.fit_transform(data['NEIGHBOURHOOD'])

# 전체 데이터셋 사용
X = data.drop('TYPE', axis=1)
y = data['TYPE']

# 데이터 타입 확인
print(X.dtypes)

# 데이터 나누기 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 훈련
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
st.subheader("분류 보고서")
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report)

# 혼동 행렬 시각화
st.subheader("혼동 행렬")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('예측된 값')
plt.ylabel('실제 값')
plt.title('혼동 행렬')
st.pyplot(fig)

# 주요 성능 지표 시각화
st.subheader("범죄 유형별 F1 점수")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=df_report.index, y=df_report['f1-score'], palette='viridis', ax=ax)
plt.xlabel('범죄 유형')
plt.ylabel('F1 점수')
plt.title('범죄 유형별 F1 점수')
plt.xticks(rotation=45)
st.pyplot(fig)

# 예제 데이터로 블록 번호와 이웃 리스트 생성 
hundred_block_list = data['HUNDRED_BLOCK'].unique().tolist() 
neighbourhood_list = data['NEIGHBOURHOOD'].unique().tolist()

# 사용자 입력을 통한 예측 
st.subheader("범죄 유형 예측") 
with st.form(key='predict_form'): 
    date = st.date_input("날짜") 
    time = st.time_input("시간") 
    hundred_block = st.selectbox("발생 위치 주소", hundred_block_list) 
    neighbourhood = st.selectbox("발생한 지역", neighbourhood_list) 
    submit_button = st.form_submit_button(label='예측하기') 
    if submit_button: 
        input_data = pd.DataFrame([[date.year, date.month, date.day, time.hour, time.minute, hundred_block, neighbourhood]], 
                                  columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'HUNDRED_BLOCK', 'NEIGHBOURHOOD']) 
        input_data['HUNDRED_BLOCK'] = label_encoder.transform(input_data['HUNDRED_BLOCK']) 
        input_data['NEIGHBOURHOOD'] = label_encoder.transform(input_data['NEIGHBOURHOOD']) 
        prediction = model.predict(input_data) 
        predicted_type = label_encoder.inverse_transform(prediction)[0] 
        st.write(f"예측된 범죄 유형은: **{predicted_type}** 입니다.")
