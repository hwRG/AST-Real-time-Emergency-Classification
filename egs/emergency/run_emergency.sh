set -x
source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

model=ast
dataset=Emergency
# full or balanced for audioset
set=full
imagenetpretrain=True
if [ $set == balanced ]
then
  bal=none
  lr=5e-5
  epoch=25
  tr_data=./data/datafiles_16k/Emergency_train_data.json
else
#  bal=bal
  bal=none
  lr=1e-5
  epoch=20
  tr_data=./data/datafiles_16k/Emergency_train_data.json
fi
te_data=./data/datafiles_16k/Emergency_eval_data.json
n_classes=19
freqm=48
timem=192
mixup=0.6
# corresponding to overlap of 6 for 16*16 patches
fstride=10
tstride=10
batch_size=36 # target_length 1024 기준 8이 20GB 
exp_dir=./exp/test-${set}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-demo

# SOLO
CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/Emergency_class_labels_indices.csv --n_class $n_classes \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain