# Variable-Hyperprior

CUDA_VISIBLE_DEVICES=0 python3 train.py  -d /home/zhan5096/Project/dataset  -e 200 -lr 1e-4 -n 8 --batch-size 16 --test-batch-size 64 --aux-learning-rate 1e-3 --patch-size 256 256 --cuda --save --seed 1926 --clip_max_norm 1.0  --stage 1 --ste 0  --loadFromPretrainedSinglemodel 0

CUDA_VISIBLE_DEVICES=0 python3 train.py  -d /home/zhan5096/Project/dataset  -e 200 -lr 1e-4 -n 8 --batch-size 16 --test-batch-size 64 --aux-learning-rate 1e-3 --patch-size 256 256 --cuda --save --seed 1926 --clip_max_norm 1.0  --stage 2 --ste 0  --refresh 1 --loadFromPretrainedSinglemodel 0 --checkpoint "path to stage1 ckpt"

CUDA_VISIBLE_DEVICES=0 python3 train.py  -d /home/zhan5096/Project/dataset  -e 200 -lr 1e-4 -n 8 --batch-size 16 --test-batch-size 64 --aux-learning-rate 1e-3 --patch-size 256 256 --cuda --save --seed 1926 --clip_max_norm 1.0  --stage 3 --ste 1 --refresh 1 --loadFromPretrainedSinglemodel 0 --checkpoint "path to stage2 ckpt"


python3 Inference.py --dataset /home/zhan5096/Project/dataset/kodak --s 11 --output_path VR -p checkpoint_best.pth.tar --patch 64 --factormode 0 --factor 0 --cuda