# Air Quality PM10 Forecast

서울 지역 공공데이터(대기 + 기상)를 이용해 다음 1시간 뒤 PM10(`PM10_1`)을 예측하는 회귀 프로젝트입니다.

## 데이터 위치

- 원본 데이터: `data/raw/air_2024.csv`
- 원본 데이터: `data/raw/air_2025.csv`
- 원본 데이터: `data/raw/weather_2024.csv`
- 원본 데이터: `data/raw/weather_2025.csv`

## 실행

```bash
python scripts/prepare_data.py
python scripts/train.py --model gbr
```

## 이번 검증 결과

- 검증 일시: 2026-03-11
- 실제 데이터 사용
- 실행 성공:
  - `python scripts/prepare_data.py --out-dir verify_processed`
  - `python scripts/train.py --model gbr --out-model verify_artifacts/model.joblib --out-metrics verify_artifacts/metrics.json --out-pred verify_artifacts/predictions.csv --out-feature-importance verify_artifacts/feature_importance.csv`
- 성능:
  - `MSE = 48.984486`
  - `R2 = 0.935329`

## 산출물

- `verify_processed/train_x.csv`
- `verify_processed/train_y.csv`
- `verify_processed/test_x.csv`
- `verify_processed/test_y.csv`
- `verify_artifacts/model.joblib`
- `verify_artifacts/metrics.json`
- `verify_artifacts/predictions.csv`
- `verify_artifacts/feature_importance.csv`

## 구현 요약

- `scripts/prepare_data.py`
  - 노트북 전처리 순서를 재현
  - 시간 파싱, merge, 결측치 처리, `PM10_lag1`, `PM10_1` 생성
- `scripts/train.py`
  - raw 데이터 또는 processed csv 모두 입력 가능
  - `lr`, `rf`, `gbr`, `xgb`, `rf_search`, `gbr_search` 지원
  - metrics와 예측값 저장
