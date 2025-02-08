## example
# bash scripts/train_policy_multi.sh [mamba_config] [task_name] [exp_name] [GPU_id]
# bash scripts/train_policy_multi.sh dp3_mamba adroit_pen 1125 0
# bash scripts/train_policy_multi.sh dp3_mamba_v2 dexart_faucet 1125 0
# bash scripts/train_policy_multi.sh dp3_mamba_hydra metaworld_stick-pull 1125 0

save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
gpu_id=${4}

cd 3D-Diffusion-Policy
export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

for seed in 0 1 2
    do
        exp_name=${task_name}-${alg_name}-${addition_info}
        run_dir="data/outputs/${exp_name}_seed${seed}"

        echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
        echo -e "\033[33mseed (to use): ${seed}\033[0m"

        wandb_mode=online
        echo -e "\033[33mTrain mode\033[0m"

        python train.py --config-name=${config_name}.yaml \
                                    task=${task_name} \
                                    hydra.run.dir=${run_dir} \
                                    training.seed=${seed} \
                                    training.device="cuda:0" \
                                    exp_name=${exp_name} \
                                    logging.mode=${wandb_mode} \
                                    checkpoint.save_ckpt=${save_ckpt}

    done
