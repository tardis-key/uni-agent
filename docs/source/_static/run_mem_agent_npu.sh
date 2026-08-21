#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient()
entrypoint = r'''
/usr/local/python3.11.15/bin/python3 -m verl.trainer.main_ppo \
  --config-name=ppo_trainer \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.num_warmup_batches=1 \
  trainer.v1.separate_async.parameter_sync_step=1 \
  transfer_queue.enable=True \
  "data.train_files=['/home/huxiaobo/data/hotpotqa/hotpotqa_train_32k.parquet']" \
  "data.val_files=['/home/huxiaobo/data/hotpotqa/hotpotqa_dev.parquet']" \
  data.prompt_key=prompt \
  data.return_raw_chat=True \
  ++data.apply_chat_template_kwargs.enable_thinking=False \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.max_prompt_length=1024 \
  data.max_response_length=256 \
  data.train_batch_size=4 \
  data.custom_cls.path=pkg://examples.mem_agent.dataset \
  data.custom_cls.name=HotpotQAMemAgentDataset \
  ++data.context_chunk_size=512 \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode=False \
  actor_rollout_ref.model.path=/home/huxiaobo/data/model/Qwen3-4B \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.prompt_length=1024 \
  actor_rollout_ref.rollout.response_length=256 \
  actor_rollout_ref.rollout.max_model_len=1280 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1280 \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.7 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
  ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=1 \
  ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=/tmp/mem_agent_smoke_logs \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=4 \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=all \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=examples/mem_agent/task_config.yaml \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=Qwen3-4B \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
  ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
  ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
  trainer.project_name=mem_agent_smoke \
  trainer.experiment_name=mem_agent_smoke_$(date +%Y%m%d_%H%M%S) \
  "trainer.logger=['console','tensorboard','rl_insight']" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.val_before_train=False \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  data.train_max_samples=4 \
  data.val_max_samples=4
'''
runtime_env = {
    "working_dir": "/home/huxiaobo/uni-agent",
    "env_vars": {
        "PYTHONPATH": "/home/huxiaobo/rl-insight:/home/huxiaobo/uni-agent:/home/huxiaobo/verl",
        "VERL_RL_INSIGHT_ENABLE": "1",
        "RL_INSIGHT_SERVER_URL": "http://127.0.0.1:18080",
        "NCCL_DEBUG": "INFO",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "RAY_DEDUP_LOGS": "0",
        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
    },
}
print(client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env))
PY
