# RL-Insight Instrumentation Guide

Uni-Agent reports business events to RL-Insight through a thin adapter. The
Agent Loop Trajectory dashboard remains framework-independent: Uni-Agent never
defines lane IDs, private dashboard metrics, or the span protocol.

## Instrumented stages

| Stage | Location | Emitted data |
|---|---|---|
| Worker initialization | `uni_agent/framework/entry.py` | Seeds verl's `RolloutTraceConfig` with trainer project/experiment names. |
| Session start | `uni_agent/framework/framework.py` | Creates one agent-loop session and attaches its immutable identity to gateway metadata and task arguments. |
| Task execution | `uni_agent/framework/task_runner.py` | One `agent_task` span with task name, sandbox image, prompt hash, reward, accuracy, completion, and reward-posting state. |
| Model generation | `uni_agent/gateway/session/session.py` | One `gateway_generation` span per gateway call, including chain/trajectory, turn, token counts, finish reason, content, tools, and errors. |
| Sandbox lifecycle | `uni_agent/sandbox/base.py` | One `agent_sandbox` span for start and stop, including provider, image, runtime ID, lifecycle, status, and error. |
| Session finish | `uni_agent/framework/framework.py` | Publishes trajectory summaries and the `agent_session` span for success, empty, or failure outcomes. |

## Architecture

```mermaid
flowchart LR
    trainer[verl trainer] --> worker[AgentFrameworkWorker]
    worker --> adapter[uni_agent.rlinsight_adapter]
    framework[Agent framework session] --> adapter
    task[Task runner] --> adapter
    gateway[Gateway generation] --> adapter
    sandbox[Sandbox lifecycle] --> adapter
    adapter --> logger[verl RLInsightLogger]
    logger --> api[rl_insight API]
    api --> tempo[Tempo traces]
    api --> prometheus[Prometheus gauges]
    tempo --> grafana[Grafana dashboard]
    prometheus --> grafana
```

Uni-Agent calls verl rather than importing trainer configuration directly. This
keeps project/experiment initialization and lazy monitor startup in the trainer
process while leaving business instrumentation in Uni-Agent.

## Training sequence

```mermaid
sequenceDiagram
    autonumber
    participant V as verl trainer
    participant W as AgentFrameworkWorker
    participant F as Agent framework
    participant G as GatewaySession
    participant T as Task runner
    participant S as Sandbox
    participant L as RLInsightLogger
    participant R as RL-Insight

    V->>W: initialize with trainer config
    W->>L: init_rollout_trace_config(config)
    F->>L: agent_loop_session(experiment, sample, session, global_steps)
    L->>R: create standard session identity
    F->>G: create session(metadata=identity)
    F->>T: run task(tools_kwargs=identity)
    T->>S: start sandbox
    S->>L: trace_sandbox_lifecycle(start)
    T->>L: task_span(result)
    G->>L: gateway_generation(turn, tokens, finish_reason)
    S->>L: trace_sandbox_lifecycle(stop)
    F->>L: session.finish(trajectories, status)
    L->>R: emit session span and hierarchy gauges
```

The exact number and order of task, generation, and sandbox spans follow the
agent's business logic. The only hard requirements are one session object per
agent session, consistent identity fields, and exactly one final `finish` call.

## Adapter API

`uni_agent/rlinsight_adapter.py` is the only Uni-Agent module that knows how to
normalize and forward completed spans.

### `init_rollout_trace_config(config)`

Reads `trainer.project_name` and `trainer.experiment_name` from the worker
configuration and initializes verl's `RolloutTraceConfig`. Call this once before
agent sessions run.

### `TaskSpanState`

Mutable state collected while a task runs. Call `record_result(result,
reward_posted=...)` after the task and reward POST complete. The context manager
reports the final `agent_task` span, including failures that propagate.

### `task_span(tools_kwargs, task_name, prompt)`

Context manager for one task. It reads the trace identity from
`tools_kwargs["_trace_identity"]`, hashes the prompt, tracks task result fields,
and reports the completed span through verl.

### `GenerationSpan`

Mutable state for one gateway generation. `success()` records normal output,
`capacity_exhausted()` records a length-exhausted generation, and `failure()`
records an exception. `report()` derives the zero-based trajectory from the
one-based chain ID and emits `gateway_generation`.

### `start_generation_span(identity)`

Creates a `GenerationSpan` with the session identity and current timestamp.
Gateway code must call `report()` in a `finally` block.

### `trace_sandbox_lifecycle(operation, sandbox, lifecycle)`

Awaitable wrapper for a sandbox start or stop coroutine. It reports
`agent_sandbox` after the operation succeeds or raises.

## Instrumentation rules

1. Do not modify `session.identity` after creation.
2. Keep `global_steps` numeric; do not stringify it.
3. Map one-based `chain_id` to zero-based `traj` through the adapter.
4. Report failed and capacity-exhausted operations, not only successes.
5. Never emit `agent_loop_*` metrics directly; `session.finish()` owns them.
6. Keep tracing best-effort: adapter failures must not break rollout.

## Verification

The integration was verified with:

| Repository | PR |
|---|---|
| RL-Insight | [#148](https://github.com/verl-project/rl-insight/pull/148) |
| Uni-Agent | [#124](https://github.com/verl-project/uni-agent/pull/124) |
| verl | [#7448](https://github.com/verl-project/verl/pull/7448) |

For `examples/mem_agent/train_mem_agent.sh`, make these local changes:

| Change | Reason |
|---|---|
| Use sibling `verl` checkout in `PYTHONPATH` | Needed for branch consistency. |
| Allow `TRAIN_FILE` override | Avoid copying the parquet. |
| Start Ray with `NPU` resources and `ASCEND_RT_VISIBLE_DEVICES` | Needed on Ascend. |
| Set `VERL_RL_INSIGHT_ENABLE=1`, `RL_INSIGHT_SERVER_URL`, and `RAY_OVERRIDE_JOB_RUNTIME_ENV=1` | Needed for RL-Insight. |
| Set `actor_rollout_ref.rollout.disable_log_stats=False` | Needed for metrics. |
| Add `rl_insight` to `trainer.logger` | Needed for logging. |

Run:

```bash
cd /home/huxiaobo
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ray start --head --resources='{"NPU":8}' --port=6293
```

```bash
REPO_ROOT=/home/huxiaobo/uni-agent \
MODEL_PATH=/home/huxiaobo/data/model/Qwen3-4B \
TRAIN_FILE=/home/huxiaobo/data/hotpotqa/hotpotqa_train_32k.parquet \
VAL_FILE=/home/huxiaobo/data/hotpotqa/hotpotqa_dev.parquet \
PYTHON_BIN=/usr/local/python3.11.15/bin/python3 \
RAY_BIN=/usr/local/python3.11.15/bin/ray \
GPU_IDS=0,1,2,3,4,5,6,7 \
PROJECT_NAME=mem_agent_smoke \
EXPERIMENT_NAME=mem_agent_smoke_$(date +%Y%m%d_%H%M%S) \
bash examples/mem_agent/train_mem_agent.sh \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  data.train_max_samples=4 \
  data.val_max_samples=4 \
  trainer.test_freq=100 \
  trainer.save_freq=100
```
