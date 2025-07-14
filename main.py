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
    
    /* 테이블 헤더 스타일 */
    .stDataFrame thead th {
        color: black !important; /* 글자색을 검정색으로 */
        font-weight: bolder !important; /* 더 굵게 */
        text-align: center !important; /* 가운데 정렬 */
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """데이터 로드 및 전처리 (로컬 파일에서 직접 로드)"""
    try:
        # 기본 파일 읽기 (다양한 인코딩 시도)
        encodings = ['cp949', 'euc-kr', 'utf-8', 'latin-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv('외국인 야간선물.csv', encoding=encoding)
                break
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        
        if df is None:
            raise ValueError("파일을 찾을 수 없거나 모든 인코딩 시도 실패")
        
        # 컬럼명 정리 (첫 번째 행이 헤더인지 확인)
        if df.iloc[0, 0] in ['단위', 'UNIT', '구분'] or '단위' in str(df.iloc[0, 0]):
            # 첫 번째 행이 단위 정보인 경우 제거
            df = df.drop(df.index[0]).reset_index(drop=True)
        
        # 두 번째 행도 헤더 정보인지 확인
        if df.iloc[0, 0] in ['', 'nan', 'NaN'] or pd.isna(df.iloc[0, 0]):
            df = df.drop(df.index[0]).reset_index(drop=True)
        
        # 컬럼명 설정
        expected_columns = ['날짜', 'K200지수', '야간선물_외국인', '정규장_외국인_선물', '정규장_외국인_현물']
        if len(df.columns) >= 5:
            df.columns = expected_columns[:len(df.columns)]
        else:
            raise ValueError(f"예상 컬럼 수(5개)와 실제 컬럼 수({len(df.columns)})가 다릅니다.")
        
        # 빈 행 제거
        df = df.dropna(subset=['날짜']).reset_index(drop=True)
        
        # 날짜 컬럼 처리
        try:
            # 날짜 형식 확인 및 변환
            df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
            
            # 날짜 변환에 실패한 행 제거
            df = df.dropna(subset=['날짜']).reset_index(drop=True)
            
            if len(df) == 0:
                raise ValueError("유효한 날짜 데이터가 없습니다.")
                
        except Exception as e:
            st.error(f"날짜 처리 중 오류: {str(e)}")
            st.info("날짜 컬럼의 첫 몇 개 값:")
            st.write(df['날짜'].head())
            raise
        # 숫자 컬럼 변환 (콤마 제거 및 숫자 변환)
        numeric_cols = ['K200지수', '야간선물_외국인', '정규장_외국인_선물', '정규장_외국인_현물']
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    # 문자열로 변환 후 콤마 제거
                    df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '')
                    # 숫자 변환 (변환 실패 시 NaN)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    st.warning(f"{col} 컬럼 변환 중 오류: {str(e)}")
        
        # NaN 값이 있는 행 제거
        df = df.dropna().reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("유효한 데이터가 없습니다.")
            
        # 데이터 정렬
        df = df.sort_values('날짜').reset_index(drop=True)
        
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
    """상관관계 분석 (나중에 사용)"""
    correlations = {}
    
    # 1. 야간선물 vs 다음날 정규장 선물
    corr1, p_value1 = stats.pearsonr(df['야간선물_외국인'], df['다음날_정규장_외국인_선물'])
    correlations['선물'] = {
        'correlation': corr1,
        'p_value': p_value1,
        'significance': '유의함' if p_value1 < 0.05 else '유의하지 않음'
    }
    
    return correlations

def create_comparison_table(df):
    """비교 표 생성"""
    # 데이터 준비
    table_data = []
    for i in range(len(df)):
        night_futures = df.iloc[i]['야간선물_외국인']
        next_day_futures = df.iloc[i]['다음날_정규장_외국인_선물']
        date = df.iloc[i]['날짜'].strftime('%Y-%m-%d')
        
        table_data.append({
            '날짜(D-Day 기준일)': date,
            '당일(D-Day) 야간선물 외국인': night_futures,
            '다음날(D+1 Day) 정규장 외국인 선물': next_day_futures
        })
    
    # 최신 날짜가 가장 위에 오도록 정렬
    table_df = pd.DataFrame(table_data).sort_values('날짜(D-Day 기준일)', ascending=False)
    
    # 스타일 적용 함수
    def style_numbers(val):
        if pd.isna(val):
            return ''
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: red; font-weight: bold;'
            elif val < 0:
                return 'color: blue; font-weight: bold;'
        return ''
    
    # 테이블 스타일 적용
    styled_df = table_df.style.applymap(
        style_numbers, 
        subset=['당일(D-Day) 야간선물 외국인', '다음날(D+1 Day) 정규장 외국인 선물']
    ).format({
        '당일(D-Day) 야간선물 외국인': '{:+,.0f}', # 양수 부호 표시
        '다음날(D+1 Day) 정규장 외국인 선물': '{:+,.0f}' # 양수 부호 표시
    })
    
    return styled_df


def main():
    # 메인 제목
    st.markdown('<h1 class="main-title">📊 외국인 야간선물 동향 분석</h1>', unsafe_allow_html=True)
    
    # 데이터 로드 (파일 업로드 없이 로컬에서 직접 로드)
    df = load_and_process_data()
    if df is None:
        st.error("데이터를 로드할 수 없습니다. '외국인 야간선물.csv' 파일이 스크립트와 같은 디렉토리에 있는지 확인해주세요.")
        return
    
    # 사이드바 (기간 선택)
    st.sidebar.header("📋 분석 옵션")
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
    
    # 첫 번째 컨텐츠: 야간선물과 다음날 정규장 선물 상관관계
    st.markdown('<h2 class="section-title">당일 외국인 야간선물 동향과 다음날 정규장 외국인 선물의 상관관계</h2>', unsafe_allow_html=True)
    
    # 비교 표
    st.markdown("### 📊 비교 표")
    comparison_table = create_comparison_table(filtered_df)
    st.dataframe(comparison_table, use_container_width=True, height=400)

    # 같은 동향을 보일 확률 계산
    if not filtered_df.empty:
        # 순매수일 때 순매수 (양수)
        same_trend_positive = ((filtered_df['야간선물_외국인'] > 0) & 
                               (filtered_df['다음날_정규장_외국인_선물'] > 0)).sum()
        
        # 순매도일 때 순매도 (음수)
        same_trend_negative = ((filtered_df['야간선물_외국인'] < 0) & 
                               (filtered_df['다음날_정규장_외국인_선물'] < 0)).sum()
        
        total_same_trend = same_trend_positive + same_trend_negative
        total_rows = len(filtered_df)

        if total_rows > 0:
            probability = (total_same_trend / total_rows) * 100
            st.markdown(f"**당일 외국인 야간선물 동향이 다음날 정규장 외국인 선물과 같은 동향을 보일 확률은 현재의 범례 기준으로 {probability:.2f}%입니다.**")
        else:
            st.info("선택된 기간에 유효한 데이터가 없습니다. 확률을 계산할 수 없습니다.")
    else:
        st.info("선택된 기간에 유효한 데이터가 없습니다. 확률을 계산할 수 없습니다.")
    
if __name__ == "__main__":
    main()
