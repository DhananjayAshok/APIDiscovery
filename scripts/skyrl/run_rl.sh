# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on a simple multiplication environment.
# uv run examples/multiply/multiply_dataset.py --output_dir $HOME/data/multiply
# export WANDB_API_KEY=<your_key_here>
# bash examples/multiply/run_multiply.sh

source configs/config.env
source setup/.venv/bin/activate

if [ -z "$storage_dir" ]; then
  echo "Error: storage_dir is not set"
  exit 1
fi
if [ -z "$trainer_policy_model" ]; then
  echo "Error: trainer_policy_model is not set"
  exit 1
fi
if [ -z "$run_name" ]; then
  echo "Warning: run_name is not set. Setting run_name to" $(basename $trainer_policy_model)
  run_name=$(basename $trainer_policy_model)
fi
if [ -z "$NUM_GPUS" ]; then
  echo "Error: NUM_GPUS is not set"
  exit 1
fi
if [ -z "$DATA_DIR" ]; then
  echo "Error: DATA_DIR is not set"
  exit 1
fi

bash scripts/warmup_rl.sh -m $trainer_policy_model
trainer_policy_model=$storage_dir/models/rl_warmup/${trainer_policy_model#*/}/final_checkpoint

trainer_ckpt_path=$storage_dir/models/rl/$run_name/ckpt
trainer_export_path=$storage_dir/models/rl/$run_name/final_checkpoint/



source $skyrl_env_dir/bin/activate || { echo "Failed to activate virtual environment at $skyrl_env_dir. Check that the path is correct and that the virtual environment is set up properly."; exit 1; }
uv pip install torch-c-dlpack-ext
cd SkyRL/skyrl-train || { echo "SkyRL/skyrl-train directory not found. Make sure the path is correct."; exit 1; }
set -x

cuda_string=""
vllm_cuda_string=""
get_all_gpus() {
    cuda_string="CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS)))"
}

# 2. Returns 2 to NUM_GPUS-1
get_gpus_from_2() {
    if [ "$NUM_GPUS" -le 2 ]; then
        echo "Error: Not enough GPUs to start from index 2" >&2
        return 1
    fi
    cuda_string="CUDA_VISIBLE_DEVICES=$(seq -s, 2 $((NUM_GPUS+1)))"
}

# 3. Returns 1 to NUM_GPUS-1
get_gpus_from_1() {
    if [ "$NUM_GPUS" -le 1 ]; then
        echo "Error: Not enough GPUs to start from index 1" >&2
        return 1
    fi
    cuda_string="CUDA_VISIBLE_DEVICES=$(seq -s, 1 $((NUM_GPUS)))"
}

if (( $NUM_GPUS == 1 )); then
    vllm_cuda_string="CUDA_VISIBLE_DEVICES=0"
    get_gpus_from_1
else
    vllm_cuda_string="CUDA_VISIBLE_DEVICES=0,1"
    get_gpus_from_2
fi

#env $vllm_cuda_string vllm serve $trainer_policy_model --dtype bfloat16 --served-model-name "model" &

#sleep 30  # Wait for the vllm server to start

#$cuda_string
# for debug, add the following:
# set NUM_GPUS=1 if feasible
# environment.skyrl_gym.max_env_workers=0 \ below
# trainer.algorithm.use_kl_loss=false change this below
# 
env HYDRA_FULL_ERROR=1 python -m examples.function_discovery.rl_main \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/val.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.max_ckpts_to_keep=3 \
  trainer.hf_save_interval=500 \
  trainer.policy.model.path=$trainer_policy_model \
  trainer.policy.model.lora.rank=16 \
  trainer.policy.model.lora.alpha=16 \
  trainer.export_path=$trainer_export_path \
  trainer.ckpt_path=$trainer_ckpt_path \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=8 \
  trainer.critic_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  +generator.engine_init_kwargs.max_model_len=4096 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.model_dtype="bfloat16" \
  generator.batched=false \
  generator.step_wise_trajectories=true \
  generator.previous_observation_only=true \
  environment.env_class=function-discovery \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="wandb" \
  trainer.project_name="function_discovery" \
  trainer.run_name="$run_name" \
  $@

 
python examples/function_discovery/adapter_to_model.py $trainer_policy_model $trainer_export_path