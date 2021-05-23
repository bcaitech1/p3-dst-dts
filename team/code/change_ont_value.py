from attrdict import AttrDict

""" ontology 바꿀 때 사용하는 함수들(사실 이거 한번 해보고 사용 안함)
"""

default_values = set(['none', 'dontcare'])

def convert_time_hour_min(s):
    if s in default_values:
        return s
    h, m = s.split(':')
    return f'시간 {h:0>2}시 {m:0>2}분'

def revert_time_hour_min(s):
    if s in default_values:
        return s

    _, h, m = s.split(' ')
    return f'{h[:2]}:{m[:2]}'

time_hour_min = AttrDict(
    convert=convert_time_hour_min,
    revert=revert_time_hour_min,
    applied=['식당-예약 시간', 
        '지하철-출발 시간', 
        '택시-도착 시간',
        '택시-출발 시간'
    ],
    example='시간 xx시 xx분'
)

def convert_hour_min(s):
    if s in default_values:
        return s
    h, m = s.split(':')
    return f'{h:0>2}시 {m:0>2}분'

def revert_hour_min(s):
    if s in default_values:
        return s

    h, m = s.split(' ')
    return f'{h[:2]}:{m[:2]}'

hour_min = AttrDict(
    convert=convert_hour_min,
    revert=revert_hour_min,
    applied=['식당-예약 시간', 
        '지하철-출발 시간', 
        '택시-도착 시간',
        '택시-출발 시간'
    ],
    example='xx시 xx분'
)
