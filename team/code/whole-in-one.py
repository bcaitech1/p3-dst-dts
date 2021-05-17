import time
import argparse
import os

from submit import submit
import parser_maker
from train import train
from inference import inference

if __name__ == "__main__":
    start = time.time()  # 시작 시간 저장
    # import train_copy
    # print("학습시간 : ", time.time()-start)
    # import inference
    # print("추론시간 : ", time.time()-start)
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('-c', '--config', 
                        type=str,
                        help="Get config file following root",
                        default='./conf.yml')
    parser = parser_maker.update_parser(parser)

    config_args = parser.parse_args()
    config_root = config_args.config
    config_args.config = None
    print(f'Using config: {config_root}')

    # task_dir = train(config_root)
    # inference(config_root, task_dir)

    test_dir='/opt/ml/p3-dst-dts/team/code/predictions'
    
    # 아래 글을 통해 자신의 key값 찾아 넣기
    # http://boostcamp.stages.ai/competitions/3/discussion/post/110
    submit("Bearer 15bdf505e0902975b2e6f578148d22136b2f7717", os.path.join(test_dir, 'final_aug_trade_10ep.csv'))

    print("전체시간 :", time.time() - start)  

 
 
