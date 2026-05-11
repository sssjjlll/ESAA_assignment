# ESAA YB 수상작 리뷰

## [주식 종료 가격 예측 경진대회]

데이터

1. **stock_list.csv : 종목 번호 데이터**
- 종목명 : 주식 종목 명
- 종목코드 : 주식 종목 코드번호
- 상장시장 : 주식 종목 상장 시장 (KOSPI or KOSDAQ)
1. **sample_submission.csv**

sample_submission 데이터

sample_submission.shape: (10,371)

코드 흐름

1. **데이터 수집**
- 한국은 경기에 민감한 국가로 이에 맞는 경기에 민감한 자산, 인덱스, 통화 데이터를 활용

1. **피처 엔지니어링**
- 여러 파생변수 피쳐 생성
- 월별 주기성 특징을 잡기 위한 sin 및 cos 변환 파생변수 추가

```jsx
day_to_sec = 24*60*60
month_to_sec = 20*day_to_sec
timestamp_s = train_x["Date"].apply(datetime.timestamp)
timestamp_freq = round((timestamp_s / month_to_sec).diff(20)[20],1)
train_x['dayofmonth_freq_sin'] = np.sin((timestamp_s / month_to_sec) * ((2 * np.pi) / timestamp_freq))
train_x['dayofmonth_freq_cos'] = np.cos((timestamp_s / month_to_sec) * ((2 * np.pi) / timestamp_freq))
```

- 이동평균 피쳐 추가
- 가격 데이터를 활용한 수익률, 고가 및 저가 범위 등 파생변수 생성
- 가격 관련 핵심 feature에 대한 모든 기업 상관계수 평균 시각화

1. **학습 및 예측**
- 모델 학습 과정에서 사용할 변수들 선언, 초기화
- 다양한 모델을 시도하기 위해 리스트에 모델 이름들을 담아 저장
- 모델 fit, predict 진행

```jsx
import pandas_datareader as pdr

# Model fit & predict
for time_ngap in range(1, target_timegap+1):
    print (F"=== Target on N+{time_ngap} ===")
    # time_ngap = 1
    
    #USD/JPY adjustment - 안전자산 선호 심리 약, 강 -> 급락, 급등 파악
    start_date_yf = "/".join([str(val_month), str(val_day - 5), str(val_year)])
    end_date_yf = "/".join([str(val_month), str(val_day), str(val_year)])
    tmp_usdjpy = pdr.DataReader("DEXJPUS", "fred", start = start_date_yf, end=end_date_yf)
    tmp_usdjpy.pct_change()
    val_adj_usdjpy = round(tmp_usdjpy.ffill().pct_change().iloc[-1],2)[0]
    
    start_date_yf = "/".join([str(test_month), sr(test_day - 5), str(test_year)])
    end_date_yf = "/".join([str(test_month), sr(test_day), str(test_year)])
    tmp_usdjpy = pdr.DataReader("DEXJPUS", "fred", start = start_date_yf, end=end_date_yf)
    test_adj_usdjpy = round(tmp_usdjpy.ffill().pct_change().iloc[-1],2)[0]
    
    for stock_name, stock_data in tqdm(stock_dic, items()):
    
        tmp_x = stock_data["df"].copy()
        tmp_y = copy.deepcopy(stock_data["target_list"][time_ngap])
        tmp_date = tmp_x["date"]
        tmp_x.drop("date", axis=1, inplace=True)
```

- 주가 급등락의 과/소적합 방지를 위해 train set에 target smoothing 적용 → 약 -2.5%의 성능향상
- 모델 피쳐 별 성능 평가 → 일자별 성능 평균인 avg와 표준편차 std를 참고해 성능이 좋은 피쳐 조합 탐색 (각 feature set 별 Linear와 XGBoost 모델 적용)

새롭게 알게 된 내용 / 어려운 점 / 배울 점

**[새롭게 알게 된 내용]**

1. 거시/외부 지표를 주가 예측에 함께 쓰는 접근
- 한국 시장이 경기 민감하다는 가정 하에, 인덱스/ 통화/ 안전자산 선호를 나타내는 지표를 피쳐로 넣어 모델이 “ 장 전체의 분위기”를 반영하게 만듦

1. 월별 주기성을 sin/cos로 인코딩하는 방법
- 날짜를 단순히 day/month로 쓰는 대신, 주기성을 연속적으로 표현하기 위해 sin/cos 변환을 추가하는 방식에 대해 새롭게 알게 됨

1. 이동평균, 수익률, 고가/저가 범위 등 가격 기반 파생변수의 중요성
- 원시 가격 자체보다, 변동성/추세/범위를 나타내는 피쳐가 예측에 더 직접적으로 기여할 수 있다는 흐름을 알게 됨

**[어려운 점]**

피쳐가 많아 거시적으로 파악 및 성능 향상의 어려움

- 이동평균/ 수익률/ 계절성/ 거시지표 등 피처의 개수가 늘어남에 따라 이해하고 파악하기에 어려움이 있었고, 피처 누락, 중복, 결측 처리방식 등에 있어 성능 향상에 영향을 줄 수 있어 보여 아리쏭했음.나

**[배울 점]**

1. 주기성은 숫자가 아닌 주기 함수로 표현하기
2. 피쳐 셋을 평균 성능 뿐 아니라 안정성 즉, 표준편차까지 함께 보기
3. 급등락이 많은 타깃에서는 스무딩/ 로버스트 학습을 고려해보기