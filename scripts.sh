nohup python main_veracity_intern.py --model_size small --repeat 1  > logs/intern_small.log 2>&1 &
nohup python main_veracity_intern.py --model_size large --repeat 1  > logs/intern_large.log 2>&1 &
nohup python main_veracity_intern.py --model_size medium --repeat 1  > logs/intern_medium.log 2>&1 &

nohup python main_veracity_llava.py --model_size small --repeat 1  > logs/llava_small.log 2>&1 &
nohup python main_veracity_llava.py --model_size large --repeat 1  > logs/ilava_large.log 2>&1 &
nohup python main_veracity_llava.py --model_size medium --repeat 1  > logs/llava_medium.log 2>&1 &