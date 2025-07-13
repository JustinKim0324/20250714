import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="외국인 야간선물 동향 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-title {
        color: #2E86AB;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-title {
        color: #A23B72;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        padding-left: 1rem;
        border-left: 4px solid #F18F01;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: white;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #E8E8E8;
        margin: 0;
    }
    
    .correlation-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .correlation-strong {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    }
    
    .correlation-moderate {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    
    .correlation-weak {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    
    .insight-box {
        background: #F8F9FA;
        border-left: 4px solid #28A745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """데이터 로드 및 전처리"""
    try:
        # CSV 파일 읽기
        df = pd.read_csv('외국인 야간선물.csv', encoding='utf-8')
        
        # 컬럼명 정리
        df.columns = ['날짜', 'K200지수', '야간선물_외국인', '정규장_외국인_선물', '정규장_외국인_현물']
        
        # 날짜 변환
        df['날짜'] = pd.to_datetime(df['날짜'])
        df = df.sort_values('날짜').reset_index(drop=True)
        
        # 숫자 컬럼 변환 (콤마 제거)
        numeric_cols = ['K200지수', '야간선물_외국인', '정규장_외국인_선물', '정규장_외국인_현물']
        for col in numeric_cols:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
        # 다음날 데이터 생성 (상관관계 분석용)
        df['다음날_K200지수'] = df['K200지수'].shift(-1)
        df['다음날_정규장_외국인_선물'] = df['정규장_외국인_선물'].shift(-1)
        df['다음날_정규장_외국인_현물'] = df['정규장_외국인_현물'].shift(-1)
        
        # K200 지수 변화율 계산
        df['K200_변화율'] = ((df['다음날_K200지수'] - df['K200지수']) / df['K200지수'] * 100).round(2)
        
        # 마지막 행 제거 (다음날 데이터가 없으므로)
        df = df[:-1]
        
        return df
    
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None

def calculate_correlation_analysis(df):
    """상관관계 분석"""
    correlations = {}
    
    # 1. 야간선물 vs 다음날 정규장 선물
    corr1, p_value1 = stats.pearsonr(df['야간선물_외국인'], df['다음날_정규장_외국인_선물'])
    correlations['선물'] = {
        'correlation': corr1,
        'p_value': p_value1,
        'significance': '유의함' if p_value1 < 0.05 else '유의하지 않음'
    }
    
    # 2. 야간선물 vs 다음날 정규장 현물
    corr2, p_value2 = stats.pearsonr(df['야간선물_외국인'], df['다음날_정규장_외국인_현물'])
    correlations['현물'] = {
        'correlation': corr2,
        'p_value': p_value2,
        'significance': '유의함' if p_value2 < 0.05 else '유의하지 않음'
    }
    
    # 3. 야간선물 vs 다음날 K200 지수 변화율
    corr3, p_value3 = stats.pearsonr(df['야간선물_외국인'], df['K200_변화율'])
    correlations['지수변화율'] = {
        'correlation': corr3,
        'p_value': p_value3,
        'significance': '유의함' if p_value3 < 0.05 else '유의하지 않음'
    }
    
    return correlations

def get_correlation_strength(corr_value):
    """상관관계 강도 분류"""
    abs_corr = abs(corr_value)
    if abs_corr >= 0.7:
        return "강한 상관관계", "correlation-strong"
    elif abs_corr >= 0.3:
        return "보통 상관관계", "correlation-moderate"
    else:
        return "약한 상관관계", "correlation-weak"

def create_correlation_chart(df, x_col, y_col, title):
    """상관관계 산점도 생성"""
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        trendline="ols",
        color_discrete_sequence=['#FF6B6B']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=16,
        title_font_color='#2E86AB',
        showlegend=False
    )
    
    fig.update_traces(
        marker=dict(size=8, opacity=0.7),
        line=dict(color='#A23B72', width=2)
    )
    
    return fig

def create_time_series_chart(df):
    """시계열 차트 생성"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('K200 지수 추이', '외국인 야간선물 거래량'),
        vertical_spacing=0.12
    )
    
    # K200 지수 추이
    fig.add_trace(
        go.Scatter(
            x=df['날짜'],
            y=df['K200지수'],
            name='K200 지수',
            line=dict(color='#2E86AB', width=2),
            fill='tonexty',
            fillcolor='rgba(46, 134, 171, 0.1)'
        ),
        row=1, col=1
    )
    
    # 외국인 야간선물 거래량
    colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in df['야간선물_외국인']]
    fig.add_trace(
        go.Bar(
            x=df['날짜'],
            y=df['야간선물_외국인'],
            name='외국인 야간선물',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        title_font_color='#2E86AB'
    )
    
    return fig

def main():
    # 메인 제목
    st.markdown('<h1 class="main-title">📊 외국인 야간선물 동향 분석</h1>', unsafe_allow_html=True)
    
    # 데이터 로드
    df = load_and_process_data()
    if df is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # 사이드바
    st.sidebar.header("📋 분석 옵션")
    
    # 기간 선택
    start_date = st.sidebar.date_input(
        "시작 날짜", 
        value=df['날짜'].min().date(),
        min_value=df['날짜'].min().date(),
        max_value=df['날짜'].max().date()
    )
    
    end_date = st.sidebar.date_input(
        "종료 날짜",
        value=df['날짜'].max().date(),
        min_value=df['날짜'].min().date(),
        max_value=df['날짜'].max().date()
    )
    
    # 데이터 필터링
    filtered_df = df[(df['날짜'].dt.date >= start_date) & (df['날짜'].dt.date <= end_date)]
    
    # 상관관계 분석
    correlations = calculate_correlation_analysis(filtered_df)
    
    # 메인 대시보드
    st.markdown('<h2 class="section-title">🎯 주요 지표</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(filtered_df)}</div>
            <div class="metric-label">분석 기간 (일)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_night_futures = filtered_df['야간선물_외국인'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_night_futures:,.0f}</div>
            <div class="metric-label">평균 야간선물 거래량</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_regular_futures = filtered_df['정규장_외국인_선물'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_regular_futures:,.0f}</div>
            <div class="metric-label">평균 정규장 선물 거래량</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_regular_spot = filtered_df['정규장_외국인_현물'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_regular_spot:,.0f}</div>
            <div class="metric-label">평균 정규장 현물 거래량</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 상관관계 분석 결과
    st.markdown('<h2 class="section-title">📈 상관관계 분석 결과</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        corr_val = correlations['선물']['correlation']
        strength, css_class = get_correlation_strength(corr_val)
        st.markdown(f"""
        <div class="correlation-box {css_class}">
            <h4>야간선물 ↔ 다음날 정규장 선물</h4>
            <h2>{corr_val:.3f}</h2>
            <p>{strength}</p>
            <p>유의성: {correlations['선물']['significance']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        corr_val = correlations['현물']['correlation']
        strength, css_class = get_correlation_strength(corr_val)
        st.markdown(f"""
        <div class="correlation-box {css_class}">
            <h4>야간선물 ↔ 다음날 정규장 현물</h4>
            <h2>{corr_val:.3f}</h2>
            <p>{strength}</p>
            <p>유의성: {correlations['현물']['significance']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        corr_val = correlations['지수변화율']['correlation']
        strength, css_class = get_correlation_strength(corr_val)
        st.markdown(f"""
        <div class="correlation-box {css_class}">
            <h4>야간선물 ↔ 다음날 K200 변화율</h4>
            <h2>{corr_val:.3f}</h2>
            <p>{strength}</p>
            <p>유의성: {correlations['지수변화율']['significance']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 시계열 차트
    st.markdown('<h2 class="section-title">📊 시계열 추이</h2>', unsafe_allow_html=True)
    time_series_chart = create_time_series_chart(filtered_df)
    st.plotly_chart(time_series_chart, use_container_width=True)
    
    # 상관관계 산점도
    st.markdown('<h2 class="section-title">🔍 상관관계 산점도</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["선물 상관관계", "현물 상관관계", "지수 변화율 상관관계"])
    
    with tab1:
        chart1 = create_correlation_chart(
            filtered_df, 
            '야간선물_외국인', 
            '다음날_정규장_외국인_선물',
            '야간선물 vs 다음날 정규장 선물'
        )
        st.plotly_chart(chart1, use_container_width=True)
    
    with tab2:
        chart2 = create_correlation_chart(
            filtered_df, 
            '야간선물_외국인', 
            '다음날_정규장_외국인_현물',
            '야간선물 vs 다음날 정규장 현물'
        )
        st.plotly_chart(chart2, use_container_width=True)
    
    with tab3:
        chart3 = create_correlation_chart(
            filtered_df, 
            '야간선물_외국인', 
            'K200_변화율',
            '야간선물 vs 다음날 K200 변화율'
        )
        st.plotly_chart(chart3, use_container_width=True)
    
    # 인사이트
    st.markdown('<h2 class="section-title">💡 주요 인사이트</h2>', unsafe_allow_html=True)
    
    insights = []
    
    # 가장 강한 상관관계 찾기
    max_corr = max(correlations.values(), key=lambda x: abs(x['correlation']))
    max_corr_name = [k for k, v in correlations.items() if v == max_corr][0]
    
    type_names = {'선물': '정규장 선물', '현물': '정규장 현물', '지수변화율': 'K200 지수 변화율'}
    
    insights.append(f"가장 강한 상관관계는 야간선물과 다음날 {type_names[max_corr_name]} 간의 관계로, 상관계수가 {max_corr['correlation']:.3f}입니다.")
    
    # 양수/음수 거래일 분석
    positive_days = len(filtered_df[filtered_df['야간선물_외국인'] > 0])
    negative_days = len(filtered_df[filtered_df['야간선물_외국인'] < 0])
    
    insights.append(f"분석 기간 중 외국인 야간선물 순매수일은 {positive_days}일, 순매도일은 {negative_days}일입니다.")
    
    # 평균 변화율
    avg_change = filtered_df['K200_변화율'].mean()
    insights.append(f"분석 기간 중 K200 지수의 평균 일일 변화율은 {avg_change:.2f}%입니다.")
    
    for insight in insights:
        st.markdown(f"""
        <div class="insight-box">
            <p>{insight}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
