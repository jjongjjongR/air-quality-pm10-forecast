DEFAULT_ENCODING = "cp949"

DROP_COLS = [
    # Location
    "지점",
    "지점명",
    "지역",
    "망",
    "측정소코드",
    "측정소명",
    "주소",
    # Time columns (kept as parsed `time`)
    "일시",
    "측정일시",
    # QC flags
    "기온 QC플래그",
    "강수량 QC플래그",
    "풍속 QC플래그",
    "풍향 QC플래그",
    "습도 QC플래그",
    "현지기압 QC플래그",
    "해면기압 QC플래그",
    "일조 QC플래그",
    "일사 QC플래그",
    "지면온도 QC플래그",
    # Too many missing values
    "적설(cm)",
    "3시간신적설(cm)",
    # Categorical codes
    "지면상태(지면상태코드)",
    "현상번호(국내식)",
    # Hard to interpret
    "운형(운형약어)",
]

TARGET_COLUMN = "PM10_1"
PM10_COLUMN = "PM10"
PM10_LAG_COLUMN = "PM10_lag1"

TIME_COLUMN = "time"
RAIN_COLUMN = "강수량(mm)"

