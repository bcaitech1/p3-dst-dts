## 사용법
`python train.py` 이러면 conf.yml에 있는 설정 기준으로 학습됩니다.

## 추론
`python inference.py --data_dir 데이터디렉토리 --model_dir 사용할모델.bin --output_dir` 출력위치

### 1. 필요한 라이브러리 설치
`pip install attrdict` <br>
`pip install PyYAML` <br>

### 2. 파일 설명
`train.py`: 기본 train.py처럼 전체적인 학습 프로세스 있음 <br>
`train_loop.py`: batch랑 model이 입력으로 제공됬을 때 loss 리턴하는 함수들 모음 <br>
>>> 모델들마다 train 부분이 달라서 이렇게 함

`preprocessor.py`: 삭제 예정<br>
`prepare_preprocessor.py`: args로부터 해당하는 preprocessor랑 model 가져오는 함수들 모음<br>
>>> 이름 변동 필요<br>

`inference.py`: 모델별 inference 함수 구현됨, 사용할 때는 기존 방식처럼 사용하면됨<br>
`evaluation.py` `eval_util.py`: 건드리지 않음<br>
`data_util.py`: 1회차 대회에 있던 seed_everything 함수 추가말고는 다른거 없음<br>
`preprocessor` 폴더: 모델별 preprocessor 정의, 사용할 하고 싶은 함수나 클래스는 `__init__.py`에 선언해서 사용하면됨<br>
`model` 폴더: 모델 정의, 사용할 하고 싶은 함수나 클래스는 `__init__.py`에 선언해서 사용하면됨<br>
`conf.yml`: 모델 돌릴 때 필요한 설정 쓰는 파일, ModelName에 해당하는 설정 사용<br>

