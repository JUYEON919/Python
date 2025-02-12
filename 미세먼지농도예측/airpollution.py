import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format

# 페이지 설정
st.set_page_config(
    page_title="서울시 미세먼지 예측",
    page_icon="🌤️",
    layout="wide"
)

# 제목
st.title("서울시 미세먼지(PM10, PM2.5) 예측 시스템")
st.write("2022년 데이터를 기반으로 2023년 1월 1일의 미세먼지를 예측합니다.")

# 데이터 로드 함수
# @st.cache_data : 데이터를 처음 로드할 때만 실제로 함수를 실행하고, 이후 동일한 함수 호출시에는 캐시된 결과를 반환
@st.cache_data
def load_data():
    df = pd.read_csv('../dataset/seoul_pm10.csv', encoding='cp949')
    df['date'] = pd.to_datetime(df['date'])
    return df

# 데이터 전처리 함수
def preprocess_data(df):
    # 결측치 처리
    df['pm10'] = df['pm10'].fillna(df['pm10'].mean())
    df['pm2.5'] = df['pm2.5'].fillna(df['pm2.5'].mean())

    # 시간 관련 피처 추가
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # 지역 원-핫 인코딩
    return pd.get_dummies(df, columns=['area'], prefix='area')
# 모델 학습 함수
def train_models(df):
    # 입력(X)과 출력(y) 정의
    X = df[['hour', 'day_of_week', 'month'] + [col for col in df.columns if col.startswith('area_')]]
    y_pm10 = df['pm10']
    y_pm25 = df['pm2.5']

    # 데이터 분할 (학습용 80%, 테스트용 20%)
    X_train, X_test, y_pm10_train, y_pm10_test = train_test_split(X, y_pm10, test_size=0.2, random_state=42)
    X_train, X_test, y_pm25_train, y_pm25_test = train_test_split(X, y_pm25, test_size=0.2, random_state=42)

    # 선형 회귀 모델 생성
    model_pm10 = LinearRegression()
    model_pm25 = LinearRegression()

    # 모델 학습
    model_pm10.fit(X_train, y_pm10_train)
    model_pm25.fit(X_train, y_pm25_train)

    return model_pm10, model_pm25

# 2023년 1월 1일 예측 데이터 생성 함수
def create_prediction_data(df, areas):
    predictions = []

    for hour in range(24):
        for area in areas:
            pred_data = {
                'hour': hour,
                'day_of_week': 6,  # 2023년 1월 1일은 일요일
                'month': 1
            }

            # 지역 원-핫 인코딩 추가
            for area_col in [col for col in df.columns if col.startswith('area_')]:
                pred_data[area_col] = 1 if area_col == f'area_{area}' else 0

            predictions.append(pred_data)

    return pd.DataFrame(predictions)

# 메인 애플리케이션
try:
    # 데이터 로드
    with st.spinner('데이터를 불러오는 중...'):
        df = load_data()

    # 기본 데이터 통계
    st.subheader("데이터 기본 정보")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("데이터 기간", f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    with col2:
        st.metric("총 관측 수", f"{len(df):,}")
    with col3:
        st.metric("지역 수", f"{df['area'].nunique()}")

    # 데이터 전처리
    with st.spinner('데이터 전처리 중...'):
        processed_df = preprocess_data(df)

    # 모델 학습
    with st.spinner('모델 학습 중...'):
        model_pm10, model_pm25 = train_models(processed_df)

    # 지역 선택
    areas = sorted(df['area'].unique())
    selected_area = st.selectbox("지역 선택", areas)

    # 2023년 1월 1일 예측
    future_data = create_prediction_data(processed_df, [selected_area])

    # 반복 예측 (PM10과 PM2.5 상호 의존성 처리)
    pm10_predictions = []
    pm25_predictions = []

    for hour in range(24):
        hour_data = future_data[future_data['hour'] == hour].copy()

        # 초기 예측에는 전체 평균 사용
        hour_data['pm2.5'] = processed_df['pm2.5'].mean()
        hour_data['pm10'] = processed_df['pm10'].mean()

        # 여러 번 반복하여 예측 정확도 향상
        for _ in range(3):
            pm10_pred = model_pm10.predict(hour_data[model_pm10.feature_names_in_])
            pm25_pred = model_pm25.predict(hour_data[model_pm25.feature_names_in_])

            hour_data['pm10'] = pm10_pred
            hour_data['pm2.5'] = pm25_pred

        pm10_predictions.append(pm10_pred[0])
        pm25_predictions.append(pm25_pred[0])

    # 예측 결과 시각화
    st.subheader(f"2023년 1월 1일 {selected_area} 미세먼지 예측")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # PM10 그래프
    ax1.plot(range(24), pm10_predictions, marker='o', linewidth=2, markersize=8)
    ax1.set_title(f'{selected_area} PM10 예측')
    ax1.set_xlabel('시간')
    ax1.set_ylabel('농도 (μg/m³)')
    ax1.grid(True)

    # PM2.5 그래프
    ax2.plot(range(24), pm25_predictions, marker='o', linewidth=2, markersize=8, color='orange')
    ax2.set_title(f'{selected_area} PM2.5 예측')
    ax2.set_xlabel('시간')
    ax2.set_ylabel('농도 (μg/m³)')
    ax2.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    # 예측값 표시
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("PM10 예측 통계")
        st.write(f"평균: {np.mean(pm10_predictions):.1f} μg/m³")
        st.write(f"최대: {np.max(pm10_predictions):.1f} μg/m³")
        st.write(f"최소: {np.min(pm10_predictions):.1f} μg/m³")

    with col2:
        st.subheader("PM2.5 예측 통계")
        st.write(f"평균: {np.mean(pm25_predictions):.1f} μg/m³")
        st.write(f"최대: {np.max(pm25_predictions):.1f} μg/m³")
        st.write(f"최소: {np.min(pm25_predictions):.1f} μg/m³")

    # 시간별 예측값 테이블
    st.subheader("시간별 예측값")
    hourly_predictions = pd.DataFrame({
        '시간': range(24),
        'PM10': [f"{x:.1f}" for x in pm10_predictions],
        'PM2.5': [f"{x:.1f}" for x in pm25_predictions]
    })
    st.dataframe(hourly_predictions, use_container_width=True)

except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
    st.error("자세한 오류 정보:")
    st.exception(e)