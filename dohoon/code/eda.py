from collections import Counter, defaultdict
from matplotlib import pyplot as plt
from typing import DefaultDict
import numpy as np
import matplotlib.font_manager as fm
import matplotlib as mpl
import os
import copy
import yaml

with open('/opt/ml/p3-dst-dts/dohoon/code/conf.yml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

conf = copy.deepcopy(conf['SharedPrams'])
#그래프 저장 장소 확인
directory=f"{conf['model_dir']}/{conf['task_name']}"
if not os.path.exists(directory):
        os.makedirs(directory)

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False
#한글 폰트로 설정하기, nanum 설치 요망
#apt-get install fonts-nanum*
plt.rcParams["font.family"] = 'NanumGothicCoding'
# print ('설정 파일 위치: ', mpl.matplotlib_fname())

#매개변수 counter에 wrong_list 혹은 correct_list가 들어올 수 있다
def get_Domain_Slot_Value_distribution_counter(counter: Counter(dict())) -> DefaultDict:
    """[key(도메인/슬롯/벨류) : value(개수) dict형식 3개 반환]

    Args:
        counter (Counter): [_evaluation에서 뽑아낸 "domain-slot-value" str배열에 counter를 씌워 개수를 센 것]

    Returns:
        dict,dict,dict: [domain,slot,value에 대한 counter]
    """
    #도메인별 개수
    domain_counter=defaultdict(int)
    #슬롯별 개수
    slot_counter=defaultdict(int)
    #벨류별 개수
    value_counter=defaultdict(int)

    for k ,v in counter.items():
        domain,slot,value=k.split('-')
        domain_counter[domain]+=v
        slot_counter[slot]+=v
        value_counter[value]+=v
        
    return domain_counter,slot_counter,value_counter

def draw_EDA(name : str, counter: dict,o_counter: dict, epoch: int):
    """[도메인, 슬롯, 벨류에 대한 정답과 오답 개수와 확률 그래프출력]

    Args:
        name (str) : [input으로 들어오는 type의 종류를 명시 ex) domain, slot, value]
        counter (dict): [getWrong_Domain_Slot_Value_distribution_counter의 오답dict 반환값]
        o_counter (dict): [getOriginal_Slot_Value_distribution_counter의 정답dict 반환값]
        epoch (int): [epoch]
    """
    counter=dict(sorted(counter.items()))
    o_counter=dict(sorted(o_counter.items()))
    
    #domain & slot EDA can run here
    if name!= "value":
        plt.figure(figsize=(22,10))
        plt.subplot(1,2,1)
        plt.title(f'wrong num per {name} ep:{epoch}')
        plt.plot(o_counter.keys(),[counter.get(slot,0) for slot in o_counter.keys()], label="wrong")
        plt.xticks(rotation=90)
        plt.plot(o_counter.keys(),o_counter.values(),label="total")
        plt.xticks(rotation=90)
        plt.legend()
        
        #도메인별 오답수/도메인별 전체 개수 를 통해 오답률을 확인할 수 있습니다.
        percentage_of_wrong=np.array([counter.get(slot,0) for slot in o_counter.keys()])/np.array(list(o_counter.values()))
        plt.subplot(1,2,2)
        plt.title(f'wrong percentage per {name} ep:{epoch}')
        plt.barh(list(o_counter.keys())[::-1],percentage_of_wrong[::-1])
        #현재 디렉토리에 사진 저장
        plt.savefig(f'{directory}/{name}_percent_barplot_ep{epoch}.png')
        plt.show()
    else : #value run here
        #오답의 개수 상위 30 종류의 value 값만 이용 
        counter=dict(sorted(counter.items(),key=lambda x:x[1],reverse=True)[:30])
        
        plt.figure(figsize=(22,10))
        plt.subplot(1,2,1)
        plt.title(f'top 30 wrong {name} num graph ep:{epoch}')
        plt.plot(counter.keys(),counter.values(), label="wrong")
        plt.xticks(rotation=90)
        plt.plot(counter.keys(),[o_counter.get(value,0) for value in counter.keys()],label="total")
        plt.xticks(rotation=90)
        plt.legend()
        percentage_of_wrong=np.array(list(counter.values()))/np.array([o_counter.get(value,0) for value in counter.keys()])
        plt.subplot(1,2,2)
        plt.title(f'wrong percentage per {name} ep:{epoch}')
        plt.barh(list(counter.keys())[::-1],percentage_of_wrong[::-1])
        #         현재 디렉토리에 사진 저장
        plt.savefig(f'{directory}/{name}_percent_barplot_ep{epoch}.png')
        plt.show()

def draw_WrongTrend(wrong_list:list(list()))-> None:
    """[오답의 추이를 도메인별, 슬롯별, 벨류별로 뽑아낸다]

    Args:
        wrong_list (list): [에폭별 오답리스트를 담은 리스트, 5에폭이라면 len(wrong_list)==5]
    """
    
    #에폭별 묶음으로 좀 더 자세히 보기 위함
    for epoch, wrong in enumerate(wrong_list):
        if epoch%3==0:
            plt.title('wrong num per domain')
        domain_counter,slot_counter,value_counter=get_Domain_Slot_Value_distribution_counter(Counter(wrong))
        domain_counter=dict(sorted(domain_counter.items()))
        plt.plot(domain_counter.keys(),domain_counter.values(), label=f"ep{epoch} domain")
        if epoch%3==0 and epoch!=0:
            plt.legend()
            plt.show()
    plt.legend()
    plt.show()

    #전체 에폭을 하나로 그린 그래프
    plt.title('wrong num per domain')
    for epoch, wrong in enumerate(wrong_list):
        domain_counter,slot_counter,value_counter=get_Domain_Slot_Value_distribution_counter(Counter(wrong))
        domain_counter=dict(sorted(domain_counter.items()))
        plt.plot(domain_counter.keys(),domain_counter.values(), label=f"ep{epoch} domain")
    plt.legend()
    plt.savefig(f'{directory}/wrong_trend.png')
    plt.show()
