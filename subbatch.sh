epoch=5
depth=5
lr=0.002
delta=0.00002
alpha=1.0

python3 train.py -g 0 --sgen results/sgen --dis results/dis --optg results/opt_g --opte results/opt_e --optd results/opt_d --epoch $epoch --depth $depth --num 5 --lr $lr --delta $delta --alpha $alpha
cp -r results results_depth$depth
cp -r img img_depth$depth

