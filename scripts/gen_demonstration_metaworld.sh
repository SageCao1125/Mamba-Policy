# bash scripts/gen_demonstration_metaworld.sh basketball
# bash scripts/gen_demonstration_metaworld.sh shelf-place
# bash scripts/gen_demonstration_metaworld.sh stick-pull
# bash scripts/gen_demonstration_metaworld.sh disassemble
# bash scripts/gen_demonstration_metaworld.sh pick-place-wall
# bash scripts/gen_demonstration_metaworld.sh push-back  
# bash scripts/gen_demonstration_metaworld.sh hand-insert    
# bash scripts/gen_demonstration_metaworld.sh pick-out-of-hole
# bash scripts/gen_demonstration_metaworld.sh assembly
# bash scripts/gen_demonstration_metaworld.sh stick-push


cd third_party/Metaworld

task_name=${1}

export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 10 \
            --root_dir "../../3D-Diffusion-Policy/data/" \
