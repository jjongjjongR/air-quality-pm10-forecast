# 공공데이터를 활용한 미세먼지(PM10) 농도 예측

서울 지역의 공공데이터(대기오염 + 기상)를 결합해 **다음 시점(t+1, 1시간 후)의 PM10(`PM10_1`)**을 예측하는 회귀(Regression) 프로젝트입니다.  
2024년 데이터를 학습, 2025년 데이터를 테스트로 사용해 **시간 순서 기반 평가**를 수행합니다.

> 원본 실습/분석은 `notebooks/AI_05반_10조.ipynb`에 있으며, 본 리포는 이를 `src/` 모듈 + `scripts/` 오케스트레이션 형태로 정리했습니다.

## 내 역할

- **조장**: 팀 관리/회의 진행/의견 조율, 최종 코드 정리 및 통합

## 폴더 구조

```
.
├─ data/raw/               # 원본 CSV (cp949)
├─ notebooks/              # 원본 노트북
├─ scripts/                # 실행(전처리/학습)
├─ src/pm10_forecast/      # 전처리/피처/학습 코드
├─ assets/                 # 결과 이미지
└─ reports/                # 발표자료/제출물(선택)
```

## 피처/타깃 정의

- 기상(weather) + 대기오염(air) 데이터를 `time` 기준으로 merge
- 결측치 처리
  - `강수량(mm)`은 0으로 대체
  - 나머지는 `ffill` → `bfill`
- 시간 피처: `month`, `day`, `hour`
- 추가 피처
  - `PM10_lag1` = 24시간 전 PM10(전일 같은 시간)
- 타깃
  - `PM10_1` = 다음 시간의 PM10(t+1)

## 사용 기술

- Python, pandas, numpy
- scikit-learn (회귀 모델/평가)
- matplotlib, seaborn (EDA/시각화)
- datetime/timedelta (시간 파싱 및 시계열 피처)
- joblib (모델 저장)

## 사용 데이터

- `data/raw/air_2024.csv`, `data/raw/air_2025.csv`: 에어코리아 서울 지역 미세먼지 관측 데이터(예: `측정일시`, `PM10`, `PM2.5` 등)
- `data/raw/weather_2024.csv`, `data/raw/weather_2025.csv`: 기상청 시간별 관측 데이터(예: `일시`, `기온(°C)`, `강수량(mm)`, `습도(%)`, `풍속(m/s)` 등)

## 결과/성과 (요약)

- 공공데이터 기반 **시계열 회귀 모델** 구현(전처리 → EDA → 피처 생성 → 모델링 → 평가)
- 단순 평균 모델 대비 **MSE 47% 감소**(팀 보고서 기준)
- `PM10_lag1`(24시간 전 PM10) 피처 추가로 예측 변동성 완화

![모델 성능](assets/모델%20성능.png)

## 트러블슈팅 (요약)

- `air`(1~24시) vs `weather`(0~23시) 시간 불일치: `측정일시`의 24시를 `00시 + 1일`로 보정(`zerofrom24` 로직)
- 결측치 다량: `강수량(mm)`은 0 대체, 나머지는 `ffill`→`bfill`
- 병합 후 불필요 컬럼/중복: `DROP_COLS`로 명시 제거

## 실행 방법 (Git Bash/WSL 기준)

### 1) 설치

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### 2) (선택) 전처리 결과 CSV 생성

노트북에서 저장하던 `train_x.csv`, `train_y.csv`, `test_x.csv`, `test_y.csv`를 생성합니다.

```bash
python scripts/prepare_data.py
```

### 3) 학습 + 평가(오케스트레이션)

```bash
python scripts/train.py --model rf
```

출력:
- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `artifacts/predictions_2025.csv`

모델 변경:

```bash
python scripts/train.py --model lr
python scripts/train.py --model gbr
# python scripts/train.py --model xgb   # xgboost 설치 필요
```

## 참고

- 원본 CSV는 `cp949` 인코딩을 사용합니다(기본값). 필요하면 `--encoding`으로 변경할 수 있습니다.
- `scripts/*.py`는 `src/`를 자동으로 `PYTHONPATH`에 추가하도록 구성되어 있어, 패키지 설치 없이 바로 실행됩니다.
- `.gitignore`에서 `*.zip`, `*.pptx`를 제외하도록 해두었습니다(필요하면 규칙을 제거하고 커밋하세요).

## 보고서

- Notion: `https://www.notion.so/290c8182fdea8009ac3ae37f1e099bc0?pvs=21`
