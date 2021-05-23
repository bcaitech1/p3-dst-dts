# 요청한 input인 gen_turn_dict의 형태는 다음과 같다
# {guid : [{turn : { text : str , state : list }}, {turn : { text : str , state : list }} ]} 

def oriNgen_dials_concat(gen_turn_dict:list,data:list)->list:
    """[원본 dials에 생성한 1개의 turn을 붙이고 뒷부분은 버리고 반환한다.]

    Args:
        gen_turn_dict (list): [dialogue의 변환한 text state들의 턴단위 리스트]
        data (list): [train_dials형식의 데이터 리스트 ]

    Returns:
        list: [턴을 변형하고 뒷부분을 제거한 train_dials형식의 dials를 반환한다.]
    """
    #gen_turn_list by gyu , data by jjj
    changed_data_list=[]
    for dialogue in data:
        d_guid=dialogue['dialogue_idx']
        if d_guid in gen_turn_dict.keys(): 
            #같은 turn이 여러개일 때를 위해 idx로 구분지어 생성한다.
            for idx, gen_turn in enumerate(gen_turn_dict[d_guid]):
                for turn,value in gen_turn.items():
                    text,state=value.values()
                    new_guid=f'{d_guid}_{idx}_{turn}'
                    new_dom=dialogue['domains']
                    new_dialogue=dialogue['dialogue'][:(turn+1)]
                    new_dialogue[-2]={'role':'user','text':text,'state':state}

                    changed_data_list.append({'dialogue_idx':new_guid,'domains':new_dom,'dialogue':new_dialogue})
                    
    return changed_data_list
            
