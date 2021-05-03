from collections import Counter, defaultdict
from matplotlib import pyplot as plt
from typing import DefaultDict
import numpy as np

Kdomain2Edomain={'관광':'tour','숙소': 'hotel','식당':'restaurant','택시':'taxi','지하철':'subway'}


def getWrong_Domain_Slot_Value_distribution_counter(wrong_counter: Counter(dict())) -> DefaultDict:
    """[오답의 이름과 개수를 쌍으로한 dict 반환]

    Args:
        wrong_counter (Counter): [_evaluation에서 뽑아낸 wrong에 counter를 씌운 것]

    Returns:
        dict,dict,dict: [domain,slot,value에 대한 오답 counter]
    """
    #도메인별 오답개수
    domain_counter=defaultdict(int)
    #슬롯별 오답개수
    slot_counter=defaultdict(int)
    #벨류별 오답개수
    value_counter=defaultdict(int)

    for k ,v in wrong_counter.items():
        domain,slot,value=k.split('-')
        domain_counter[domain]+=v
        slot_counter[slot]+=v
        value_counter[value]+=v
        
    #주석을 풀면 draw_WrongTrend에서도 매번 출력됨
    # print(sorted(domain_counter.items(),key=lambda x : x[1], reverse=True))
    # print(sorted(slot_counter.items(),key=lambda x : x[1], reverse=True))
    # print(sorted(value_counter.items(),key=lambda x : x[1], reverse=True)[:5])

    #관광,숙소,식당이 제일 많은데 관광이 제일 적다. 
    #택시가 관광과 식당보다는 적은데 1900인 정보는 유의미하다
    #value측면에서 doncare, no ,yes가 1,2,3등인걸 보니 역시 boolean을 잘 못맞춘다.
    return domain_counter,slot_counter,value_counter

def getOriginal_Slot_Value_distribution_counter(correct_counter : dict) -> DefaultDict:
    """[원본의 분포 확인]

    Args:
        dev_labels (dict): [해당 에폭에서 train_labels인 정답 라벨, _evaluation에서 뽑아낸 correct에 counter를 씌운 것]

    Returns:
        dict,dict,dict: [domain,slot,value정답에 대해 분포 counter]
    """
    #원본의 도메인,벨류 개수
    o_domain_counter=defaultdict(int) #original domain counter
    o_slot_counter=defaultdict(int) #original slot counter
    o_value_counter=defaultdict(int) #original value counter

    for k,v in correct_counter.items():
        domain,slot,value=k.split('-')
        o_domain_counter[domain]+=v
        o_slot_counter[slot]+=v
        o_value_counter[value]+=v
        
#     print(sorted(o_domain_counter.items(),key=lambda x : x[1], reverse=True)[:5])
#     print(sorted(o_slot_counter.items(),key=lambda x : x[1], reverse=True)[:5])
#     print(sorted(o_value_counter.items(),key=lambda x : x[1], reverse=True)[:5])
    #value 중 1,2는 숙박, 음식 부문에서 "예약기간", "예약 명수"를 포함한다 
    return o_domain_counter,o_slot_counter,o_value_counter

def draw_EDA(domain_counter: dict,o_domain_counter: dict, epoch: int):
    """[5가지 도메인에 대한 정답과 오답 그래프 출력]

    Args:
        domain_counter (dict): [getWrong_Domain_Slot_Value_distribution_counter의 첫번째 반환값]
        o_domain_counter (dict): [getOriginal_Slot_Value_distribution_counter의 첫번째 반환값]
        epoch (int): [epoch]
    """
    domain_counter=dict(sorted(domain_counter.items()))
    o_domain_counter=dict(sorted(o_domain_counter.items()))
    
    plt.title(f'wrong num per domain ep:{epoch}')
    
    plt.plot([Kdomain2Edomain[d] for d in domain_counter.keys()],domain_counter.values(), label="wrong")
    #plt.plot([Kdomain2Edomain[d] for d in o_domain_counter.keys()],o_domain_counter.values(),label="total")
    plt.legend()
    #현재 디렉토리에 사진 저장
    plt.savefig(f'domain_graph_plot_ep{epoch}.png')
    plt.show()
    plt.clf()
    #도메인별 오답수/도메인별 전체 개수 를 통해 오답률을 확인할 수 있습니다.
    percentage_of_wrong=np.array(list(domain_counter.values()))/np.array(list(o_domain_counter.values()))
    plt.title(f'wrong percentage per domain ep:{epoch}')
    plt.bar([Kdomain2Edomain[d] for d in o_domain_counter.keys()],percentage_of_wrong)
    plt.savefig(f'domain_percent_barplot_ep{epoch}.png')
    plt.show()
    #슬롯과 벨류도 하고 싶은데 한글과의 호환이 되지 않기 때문에..심지어 한글과의 호환은 개인별 노트북환경에서 설정해야함
    
def draw_WrongTrend(wrong_list:list(list()))-> None:
    """[오답의 추이를 도메인별, 슬롯별, 벨류별로 뽑아낸다]

    Args:
        wrong_list (list): [에폭별 오답리스트를 담은 리스트, 5에폭이라면 len(wrong_list)==5]
    """
    
    #에폭별 묶음으로 좀 더 자세히 보기 위함
    for epoch, wrong in enumerate(wrong_list):
        if epoch%3==0:
            plt.title('wrong num per domain')
        domain_counter,slot_counter,value_counter=getWrong_Domain_Slot_Value_distribution_counter(Counter(wrong))
        domain_counter=dict(sorted(domain_counter.items()))
        plt.plot([Kdomain2Edomain[d] for d in domain_counter.keys()],domain_counter.values(), label=f"ep{epoch} domain")
        if epoch%3==0 and epoch!=0:
            plt.legend()
            plt.show()
    plt.legend()
    plt.show()

    #전체 에폭을 하나로 그린 그래프
    plt.title('wrong num per domain')
    for epoch, wrong in enumerate(wrong_list):
        domain_counter,slot_counter,value_counter=getWrong_Domain_Slot_Value_distribution_counter(Counter(wrong))
        domain_counter=dict(sorted(domain_counter.items()))
        plt.plot([Kdomain2Edomain[d] for d in domain_counter.keys()],domain_counter.values(), label=f"ep{epoch} domain")
    plt.legend()
    plt.savefig('wrong_trend.png')
    plt.show()
