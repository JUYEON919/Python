import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format

# 데이터 로드
file_path = 'dataset/crime.csv'
crime_data = pd.read_csv(file_path)

# 결측치 및 중복값 통계
missing_stats = crime_data.isnull().sum()
duplicate_count = crime_data.duplicated().sum()
st.write("### 결측치 통계")
st.write(missing_stats)
st.write(f"### 중복값 개수: {duplicate_count}")

# 결측치 처리: 주요 열에서 결측치를 가진 행 제거
required_columns = ['TYPE', 'NEIGHBOURHOOD', 'HOUR', 'MINUTE']
missing_counts = crime_data[required_columns].isnull().sum()
st.write("### 주요 열 결측치 확인")
st.write(missing_counts)

# 결측치가 있는 행 제거
crime_data = crime_data.dropna(subset=required_columns)

# 시간대 범주 생성
crime_data['HOUR'] = crime_data['HOUR'].astype(float)  # HOUR 데이터 형식 보정
crime_data['TIME_BLOCK'] = pd.cut(
    crime_data['HOUR'], 
    bins=[0, 6, 12, 18, 24], 
    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
    include_lowest=True
)

# 주말 여부 계산
crime_data['IS_WEEKEND'] = (crime_data['DAY'] % 7 >= 5).astype(int)

# 중복값 제거
#st.write(f"### 중복값 제거 전 데이터 크기: {crime_data.shape}")
crime_data.drop_duplicates(inplace=True)
#st.write(f"### 중복값 제거 후 데이터 크기: {crime_data.shape}")

# 범주형 데이터 인코딩
label_encoders = {}
for col in ['TIME_BLOCK', 'NEIGHBOURHOOD']:
    le = LabelEncoder()
    crime_data[col] = le.fit_transform(crime_data[col].astype(str))
    label_encoders[col] = le

# TYPE 레이블 인코딩
le_target = LabelEncoder()
crime_data['TYPE'] = le_target.fit_transform(crime_data['TYPE'].astype(str))

# 결과 확인
st.write("### 전처리된 데이터 샘플")
st.write(crime_data.head())

# 데이터 준비
features = ['YEAR', 'MONTH', 'DAY', 'TIME_BLOCK', 'IS_WEEKEND', 'NEIGHBOURHOOD']
target = 'TYPE'

X = crime_data[features]
y = crime_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)  # 학습 단계

# 예측 및 성능 평가
y_pred = model.predict(X_test)
#st.write(f"### 테스트 데이터 예측 완료")
#st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
#st.write("Classification Report:")
#st.text(classification_report(y_test, y_pred))

# 예측 결과와 실제 결과 비교
comparison = pd.DataFrame({
    'Date': pd.to_datetime(X_test[['YEAR', 'MONTH', 'DAY']].astype(str).agg('-'.join, axis=1)),
    'Actual': le_target.inverse_transform(y_test),
    'Predicted': le_target.inverse_transform(y_pred)
})
comparison['Match'] = comparison['Actual'] == comparison['Predicted']
st.write("### 예측값과 실제값 비교")
st.write(comparison)

# 일자별 요약 생성
daily_summary = comparison.groupby('Date').agg(
    Accuracy=('Match', 'mean'),
    Count=('Match', 'size')
).reset_index()
st.write("### 일자별 예측 요약")
st.write(daily_summary)

# Streamlit 셀렉트 박스 UI 구성
st.title("범죄 유형 예측")
selected_year = st.selectbox("년도 선택", sorted(crime_data['YEAR'].unique()))
selected_month = st.selectbox("월 선택", sorted(crime_data['MONTH'].unique()))
selected_day = st.selectbox("일 선택", sorted(crime_data['DAY'].unique()))
selected_time_block = st.selectbox("시간대 선택", ['Night', 'Morning', 'Afternoon', 'Evening'])
selected_neighbourhood = st.selectbox("지역 선택", label_encoders['NEIGHBOURHOOD'].classes_)

# 선택된 입력 데이터 처리
input_data = pd.DataFrame([{
    'YEAR': selected_year,
    'MONTH': selected_month,
    'DAY': selected_day,
    'TIME_BLOCK': label_encoders['TIME_BLOCK'].transform([selected_time_block])[0],
    'IS_WEEKEND': selected_day % 7 >= 5,
    'NEIGHBOURHOOD': label_encoders['NEIGHBOURHOOD'].transform([selected_neighbourhood])[0]
}])

# 예측
predicted_type = model.predict(input_data)
predicted_type_label = le_target.inverse_transform(predicted_type)[0]
st.write("### 예측 결과")
st.write(f"선택된 조건에서 예상 범죄 유형: {predicted_type_label}")
