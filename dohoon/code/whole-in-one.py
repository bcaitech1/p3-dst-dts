import time
from submit import submit
from train_copy import train
from inference import inference
import os
if __name__ == "__main__":
    start = time.time()  # 시작 시간 저장
    # import train_copy
    # print("학습시간 : ", time.time()-start)
    # import inference
    # print("추론시간 : ", time.time()-start)
    config_name='/opt/ml/p3-dst-dts/dohoon/code/conf.yml'
    train(config_name)
    inference(config_name)

    test_dir='/opt/ml/p3-dst-dts/dohoon/code/predictions'
    
    # 아래 글을 통해 자신의 key값 찾아 넣기
    # http://boostcamp.stages.ai/competitions/3/discussion/post/110
    # submit("Bearer 15bdf505e0902975b2e6f578148d22136b2f7717", os.path.join(test_dir, 'predictions_trade.csv'))

    print("전체시간 :", time.time() - start)  

 
 
