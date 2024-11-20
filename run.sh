expath=/mnt/bn/yongjing-yg

export TRANSFORMERS_CACHE=${expath}/hfcache
export HF_HOME=${expath}/hfcache
export HF_DATASETS_OFFLINE=1
# export OMP_NUM_THREADS=8
export WANDB_PROJECT="semformer_ml"

export NLTK_DATA=${expath}/nltk_data

llmtype="Qwen2-0.5B"

modelpath=${expath}/hfmodels/$llmtype

cnnf=$expath/mydata/wmt19_3langs

write_path=${expath}/semml_result

# pip install transformers==4.44.0
# pip install pytest==8.0.0
# pip install evaluate
# pip install --upgrade accelerate
# pip install deepspeed==0.14.1
# pip install -U byted-wandb -i https://bytedpypi.byted.org/simple

scrip_dir=./

# gpu=0
# export CUDA_VISIBLE_DEVICES=0

block_size=512

ztokens=8 # 16 32 64
alpha=0.5 # default=1

beta=1 # default=1
zdim=64 # 32 64 128
encstr=suffix

lr=7e-5

znorm=0

# total batch:1M, 8gpu * 128 * 1024
# close dropout in config.json

# total batch:1M, 8gpu * 128*4 * 256

# bz 128: step 
bs=4
wd=0.1
acc=8

ae_min_length=32

load_pretrained_enc_and_fix=1

saeact=4096
# saeact=16384
# sparsity_coefficient=6e-4
sparsity_coefficient=5e-5

modelname=${llmtype}_wmt_stdaez${ztokens}dim${zdim}lr${lr}a${alpha}b${beta}orth10lr${lr}

# ptae=$write_path/checkpoints/$modelname
# beta=0
# alpha=0.1

# lr=2e-5
predictor="linear"
regloss="l2"

sae_kl_temperature=0.7

# modelname=${modelname}_trainlma${alpha}pred${predictor}reg${regloss}

cmd="$scrip_dir/run_aereg_ml_mt.py"
# --use_flash_attention_2 \
    # --zero3_save_16bit_model True \
    # --do_eval --evaluation_strategy steps --eval_steps 2000 \
    # --load_best_model_at_end \
    # --overwrite_output_dir \
    # --num_processes $(($ARNOLD_WORKER_NUM * $ARNOLD_WORKER_GPU)) \
    # --streaming --max_steps 15822 \
    # --streaming --max_steps 15822 \

accelerate launch \
    --num_processes $(($ARNOLD_WORKER_NUM * $ARNOLD_WORKER_GPU)) \
    --num_machines $ARNOLD_WORKER_NUM \
    --machine_rank $ARNOLD_ID \
    --main_process_ip $ARNOLD_WORKER_0_HOST \
    --main_process_port $ARNOLD_WORKER_0_PORT \
    --use_deepspeed \
    --zero_stage 1 \
    --deepspeed_multinode_launcher standard \
    $cmd \
    --model_name_or_path $modelpath \
    --overwrite_output_dir \
    --num_train_epochs 2 \
    --open_dp True \
    --use_sae False --zorth_reg 10 \
    --sae_architecture standard --sae_use_pre_enc_bias True --sae_kl_sparse 0 \
    --sae_activation_size $saeact --sae_bandwidth 0.001 --sae_sparsity_coefficient $sparsity_coefficient \
    --regloss $regloss --sae_kl_temperature $sae_kl_temperature \
    --ae_min_length $ae_min_length \
    --only_tunelayers 1 --group_doc False --from_scratch False \
    --dataset_name $cnnf --from_disk $cnnf \
    --preprocessing_num_workers 64 \
    --report_to none --run_name $modelname \
    --ztokens $ztokens --encstr $encstr --z_from_layer -4 --predictor $predictor \
    --share_emb_decoders 0 \
    --load_pretrained_enc_and_fix $load_pretrained_enc_and_fix \
    --alpha $alpha --beta $beta \
    --zdim $zdim --znorm $znorm \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps $acc \
    --logging_steps 50 \
    --learning_rate $lr --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --seed 42 \
    --bf16 \
    --block_size $block_size \
    --save_total_limit -1 --save_strategy epoch --save_steps 5000 \
    --do_train \
    --output_dir $write_path/checkpoints/$modelname \
     >$write_path/logs/log.$modelname 2>&1

