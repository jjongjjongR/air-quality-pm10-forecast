# 미세먼지(PM10) 농도 예측 (t+1, 1시간 후)

2024년 데이터로 학습하고 2025년 데이터로 평가하여, **1시간 뒤 PM10 농도(`PM10_1`)**를 예측합니다.  
원본 노트북(`notebooks/AI_05반_10조.ipynb`)의 내용을 `src/`(모듈) + `scripts/`(오케스트레이션) 형태로 정리했습니다.

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
