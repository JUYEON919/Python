import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
import time  # 시간 측정을 위한 모듈

#todo: 스케일링 추가해보기

# Streamlit 캐시 설정 -> 데이터 로드와 모델 학습 시간을 단축
@st.cache_data                          # 데이터 로드 캐싱
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_resource                        # 모델 학습 캐싱
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# 폰트 및 그래프 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
pd.options.display.float_format = '{:.2f}'.format


def load_data(file_path):
    return pd.read_csv(file_path)

# 데이터 로드
file_path = 'dataset/crime.csv'
crime_data = pd.read_csv(file_path)

#제목
st.title("범죄 유형 예측")
st.write("2003년 ~ 2017년 데이터를 기반으로 범죄 예측합니다.")

# 결측치 및 중복값 제거
required_columns = ['TYPE', 'NEIGHBOURHOOD', 'HOUR', 'MINUTE']
crime_data = crime_data.dropna(subset=required_columns).drop_duplicates()

# 주말 여부 계산 (0: 평일, 1: 주말)
crime_data['IS_WEEKEND'] = (crime_data['DAY'] % 7 >= 5).astype(int)

# 범주형 데이터 인코딩
label_encoders = {}
for col in ['NEIGHBOURHOOD']:
    le = LabelEncoder()
    crime_data[col] = le.fit_transform(crime_data[col].astype(str))
    label_encoders[col] = le

# TYPE 레이블 인코딩
le_target = LabelEncoder()
crime_data['TYPE'] = le_target.fit_transform(crime_data['TYPE'].astype(str))

# 시간대 범주화 함수
def categorize_time(hour):
    if 0 <= hour < 6:
        return '밤'
    elif 6 <= hour < 12:
        return '아침'
    elif 12 <= hour < 18:
        return '낮'
    else:
        return '오후'
crime_data['TIME_OF_DAY'] = crime_data['HOUR'].apply(categorize_time)

# 시간대 인코딩
time_of_day_encoder = LabelEncoder()
crime_data['TIME_OF_DAY'] = time_of_day_encoder.fit_transform(crime_data['TIME_OF_DAY'])

# 데이터 준비
features = ['YEAR', 'MONTH', 'DAY', 'TIME_OF_DAY', 'IS_WEEKEND', 'NEIGHBOURHOOD']
target = 'TYPE'
X = crime_data[features]
y = crime_data[target]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model = train_model(X_train, y_train)

# 예측 및 성능 평가
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred, average='weighted')

# Streamlit 화면에 정밀도 표시
#st.write(f"모델의 정밀도(Precision): {precision:.2f}")

# 범죄 유형 숫자를 범죄 이름으로 변환
crime_data['TYPE'] = le_target.inverse_transform(crime_data['TYPE'])

# Streamlit 셀렉트 박스 UI 구성
with st.form(key='predict_form'):
    selected_year = st.selectbox("년도 선택", sorted(crime_data['YEAR'].unique()), format_func=lambda x: f"{x}년")
    selected_month = st.selectbox("월 선택", sorted(crime_data['MONTH'].unique()), format_func=lambda x: f"{x}월")
    selected_day = st.selectbox("일 선택", sorted(crime_data['DAY'].unique()), format_func=lambda x: f"{x}일")
    selected_time_of_day = st.selectbox("시간대 선택", ['아침', '낮', '오후', '밤'])
    selected_neighbourhood = st.selectbox("지역 선택", label_encoders['NEIGHBOURHOOD'].classes_)
    
    if st.form_submit_button(label='예측하기'):
        # 선택된 입력 데이터 처리
        input_data = pd.DataFrame([{
            'YEAR': selected_year,
            'MONTH': selected_month,
            'DAY': selected_day,
            'TIME_OF_DAY': time_of_day_encoder.transform([selected_time_of_day])[0],
            'IS_WEEKEND': selected_day % 7 >= 5,
            'NEIGHBOURHOOD': label_encoders['NEIGHBOURHOOD'].transform([selected_neighbourhood])[0]
        }])

        # 예측
        predicted_type = model.predict(input_data)
        predicted_type_label = le_target.inverse_transform(predicted_type)[0]
        st.write("### 예측 결과")
        st.write(f"선택된 조건에서 예상 범죄 유형: {predicted_type_label}")
 
 # 월별 및 일별 범죄량 집계
monthly_crime_count = crime_data.groupby(['YEAR', 'MONTH', 'TYPE']).size().reset_index(name='Count')
daily_crime_count = crime_data.groupby(['YEAR', 'MONTH', 'DAY', 'TYPE']).size().reset_index(name='Count')

# 범죄량 시각화 함수
def plot_crime_counts(data, x_col, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=data, x=x_col, y='Count', hue='TYPE', marker='o', ax=ax)
    ax.set_title(title)
    ax.legend(title='TYPE', bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig

# 월별 범죄량 시각화
st.write("### 월별 범죄 유형에 따른 범죄량")
st.pyplot(plot_crime_counts(monthly_crime_count, 'MONTH', '월별 범죄 유형에 따른 범죄량'))

# 일별 범죄량 시각화
st.write("### 일별 범죄 유형에 따른 범죄량")
st.pyplot(plot_crime_counts(daily_crime_count, 'DAY', '일별 범죄 유형에 따른 범죄량'))

################################################

# 예측값과 실제값 비교 데이터프레임 생성
comparison_df = pd.DataFrame({
    'Actual': le_target.inverse_transform(y_test),
    'Predicted': le_target.inverse_transform(y_pred)
})

# 실제값과 예측값의 개수 집계
actual_counts = comparison_df['Actual'].value_counts().sort_index()
predicted_counts = comparison_df['Predicted'].value_counts().sort_index()

# 범죄 유형별 개수를 하나의 데이터프레임으로 합치기
comparison_counts = pd.DataFrame({
    'Actual': actual_counts,
    'Predicted': predicted_counts
}).fillna(0)

# 선그래프 시각화
def plot_comparison_line(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    data.index = data.index.astype(str)  # 범죄 유형을 문자열로 변환
    data.plot(kind='line', marker='o', ax=ax)
    ax.set_title('예측값과 실제값 비교 (범죄 유형별)')
    ax.set_xlabel('범죄 유형')
    ax.set_ylabel('건수')
    ax.legend(title='데이터 유형')
    plt.xticks(rotation=45)
    return fig

# Streamlit 화면에 그래프 출력
st.write("### 예측값과 실제값 비교 (선그래프)")
st.pyplot(plot_comparison_line(comparison_counts)) 

#######################################################
        
# 데이터 로드 시간 측정
start_time = time.time()
crime_data = load_data(file_path)
st.write(f"데이터 로드 시간: {time.time() - start_time:.2f}초")
