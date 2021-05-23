## 사용법

`python train.py -c config_파일_이름` 으로 학습 가능

`python inference.py -t 모델_파일_위치` 로 prediction 생성 가능

`python whole-in-one.py` : 미리 정의한 config파일을 통해 학습부터 제출까지 (학습-추론-제출)의 프로세스를 한번에 수행할 수 있게 script화한 one-click 파일

### 1. 파일 설명

> 메인 학습 파일

`train.py` : 학습 전체 프로세스가 있는 파일

> 모델별 작성된 파일들/폴더

`model` : 각각 모델이 작성된 폴더

`preprocessor`  : 각각 모델별로 필요한 preprocessor 정의된 폴더, 

`train_loop.py` : 모델 별로 하나의 batch를 어떻게 train시킬지 정의되어있는 파일

`losses.py`: 모델별 필요한 loss 정의된 파일

`inference.py` : 모델별 eval loader로 어떻게 inference 할지 정의된 파일, 또 이 파일 스크립트로 실행해서 prediction을 구할 수 있음

> EDA 와 Augmentation 파일

`aug_utils.py` : 숙소/택시/지하철 도메인의 "시간" slot-value를 변경하면서 다양한 시간 표현으로 만들어 주기 위함 (ex. 15시 30분 → 낮 3시 30분 / 오후 3시 반 / 15:30 / etc..)

`augmentation.py` : aug_utils를 이용하여 augmentation 파일을 만들어낼 때 사용하는 실행 파일

`eda.py` : 프로젝트의 방향성을 잡기 위해 모델이 어느 부분을 틀리는지 그래프로 확인하는 파일들을 과 틀린 value를 dialogue와 함께 보기 위한 파일들을 포함. domain / domain-slot / slot / value 의 전체 데이터 대비 오답 빈도와 오답률을 그래프로 그리는 파일 존재. dialogue id 별로 trun들에 대해 오답과 정답 value를 동시에 포함하는 파일을 생성하며 dialogue id로 궁금한 dialogue 내 오답 value와 text를 확인할 수 있음 

> 편리함을 위한 파일

`submit.py` : csv파일을 다운받아 서버에 제출할 필요없이 원격 저장소에 있는 파일로 바로 제출할 수 있게 해주는 파일

`whole-in-one.py` : 학습부터 제출까지 (학습-추론-제출)의 프로세스를 한번에 수행할 수 있게 script화한 one-click 파일

> Eval 관련 파일들

`eval_utils.py`: 기본 제공된 파일

`evaluation.py` : 기본 제공된 파일, 조금 바뀜

> 데이터나 학습전 준비하는 파일들

`data_utils.py` : 데이터 관련 제공된 파일, 거의 바뀐거 없음

`prepare.py` : config로부터 각종 필요한 것들 만드는 파일

`change_ont_value.py` : ontology 바꿀때 사용하는 파일(사용 안함)

> wandb 관련 파일

`wandb_stuff.py` : wandb 연동을 위해 필요한 함수들, `wandb.init`이나 logging 관련 함수들이 존재

`parser_maker.py`: wandb sweep 연동을 위해 작성한 함수들,  sweep에 필요한 argparser option 생성해줌, 후반에 사용 안함

`sweep.yml` :  wandb sweep 용 config 파일

> 잡다한 파일들

`training_recorder.py` : running loss 계산기

`conf.yml`: config 파일

`requirements.txt` : 중간에 한번 생성해서 실제 모든 필요한 library 없을 수도 있어요.

`README.md` : 이 파일

> 안 쓰는 파일들

`train_copy.py`

`train_inference.py`

`test.ipynb`

`base_config.yml`
