"""Small rl-insight trace adapter shared by uni-agent instrumentation points.

The public reporting path is intentionally narrow: callers collect a completed
span's attributes and timestamps, and this module normalizes values to the
OpenTelemetry scalar types expected by rl-insight.

The module also owns a ``contextvars`` trace identity for task-local
instrumentation. Agent-loop session identity and dashboard metadata are owned
by rl-insight.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf
from rl_insight.agent_loop import agent_loop_lane_id
from verl.utils.rollout_trace import RolloutTraceConfig
from verl.utils.tracking import RLInsightLogger

logger = logging.getLogger(__name__)

_trace_identity: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "uni_agent_trace_identity",
    default=None,
)


def _normalize_value(value: Any) -> str | bool | int | float:
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _normalize_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Return an OTel-safe attribute mapping.

    Complex values such as dict/list are JSON-encoded. ``None`` is represented
    as an empty string so consumers can query on the key without nullable
    attribute handling.
    """
    return {str(key): _normalize_value(value) for key, value in (attributes or {}).items()}


def _set_trace_identity(identity: dict[str, Any] | None) -> contextvars.Token[dict[str, Any] | None]:
    """Publish trace identity for the current async/thread context."""
    return _trace_identity.set(dict(identity or {}))


def _reset_trace_identity(token: contextvars.Token[dict[str, Any] | None]) -> None:
    """Restore the previous trace identity."""
    _trace_identity.reset(token)


def _get_trace_identity() -> dict[str, Any]:
    """Return a copy of the current context's trace identity."""
    return dict(_trace_identity.get() or {})


def _start_span() -> int:
    """Return the current Unix epoch time in nanoseconds."""
    return time.time_ns()


def _report_span(
    name: str,
    *,
    start_time_ns: int,
    attributes: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
) -> None:
    """Report one completed span through verl's RLInsightLogger adapter.

    Common trace labels from the current context are merged first, so callers
    only need to add span-specific attributes.
    """
    merged_attributes = {**(identity or _get_trace_identity()), **(attributes or {})}
    try:
        RLInsightLogger.trace_span(
            name=name,
            start_time_ns=start_time_ns,
            end_time_ns=time.time_ns(),
            attributes=_normalize_attributes(merged_attributes),
        )
    except Exception:  # noqa: BLE001 - tracing must never break the rollout
        logger.exception("failed to report rl-insight span %s", name)


def init_rollout_trace_config(config: Any) -> None:
    """Initialize rollout trace identity from the trainer config."""
    RolloutTraceConfig.init(
        project_name=OmegaConf.select(config, "trainer.project_name", default=None),
        experiment_name=OmegaConf.select(config, "trainer.experiment_name", default=None),
        backend=None,
    )


@dataclass
class TaskSpanState:
    """Mutable task result collected by :func:`task_span`."""

    start_ns: int
    task_name: str
    image_ref: Any
    prompt_hash: str
    status: str = "success"
    error: str | None = None
    reward: Any = None
    accuracy: Any = None
    finished: Any = None
    reward_posted: bool = False

    def record_result(self, result: Any, *, reward_posted: bool) -> None:
        self.reward = result.reward
        self.accuracy = result.accuracy
        self.finished = result.finished
        self.reward_posted = reward_posted

    def _attributes(self) -> dict[str, Any]:
        return {
            "monitor.trace_source": "task",
            "task_name": self.task_name,
            "image_ref": self.image_ref,
            "prompt_hash": self.prompt_hash,
            "status": self.status,
            "reward": self.reward,
            "accuracy": self.accuracy,
            "finished": self.finished,
            "reward_posted": self.reward_posted,
            "error": self.error,
        }


@contextmanager
def task_span(
    tools_kwargs: dict[str, Any] | None,
    *,
    task_name: str,
    prompt: Any,
):
    """Bind task identity and report one completed task span."""
    task = tools_kwargs.get("task") if tools_kwargs else {}
    sandbox = task.get("sandbox", {}) if isinstance(task, dict) else {}
    prompt_hash = hashlib.sha256(
        json.dumps(prompt, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    identity = dict((tools_kwargs or {}).get("_trace_identity") or {})
    token = _set_trace_identity(identity)
    state = TaskSpanState(
        start_ns=_start_span(),
        task_name=task_name,
        image_ref=sandbox.get("image"),
        prompt_hash=prompt_hash,
    )
    try:
        yield state
    except BaseException as exc:
        state.status = "failure"
        state.error = f"{type(exc).__name__}:{exc}"
        raise
    finally:
        try:
            _report_span(
                name="agent_task",
                start_time_ns=state.start_ns,
                attributes=state._attributes(),
            )
        finally:
            _reset_trace_identity(token)


@dataclass
class GenerationSpan:
    """Mutable gateway-generation span state."""

    identity: dict[str, Any]
    start_ns: int
    status: str = "success"
    error: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    chain_id: int | None = None
    turn: int | None = None
    type: str | None = None
    tools: list[str] | None = None
    content: str | None = None

    def capacity_exhausted(self, *, prompt_tokens: int, chain_id: int | None) -> None:
        self.status = "capacity_exhausted"
        self.finish_reason = "length"
        self.prompt_tokens = prompt_tokens
        self.chain_id = chain_id

    def success(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        chain_id: int | None,
        turn: int,
        assistant_msg: dict[str, Any],
        finish_reason: str | None,
    ) -> None:
        tool_calls = assistant_msg.get("tool_calls") or []
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.chain_id = chain_id
        self.turn = turn
        self.type = "tool" if tool_calls else "llm"
        self.tools = [tool_call.get("function", {}).get("name", "") for tool_call in tool_calls]
        self.content = str(assistant_msg.get("content") or "")[:500]
        self.finish_reason = finish_reason

    def failure(self, exc: BaseException) -> None:
        self.status = "error"
        self.error = f"{type(exc).__name__}:{exc}"

    def report(self) -> None:
        traj = str(self.chain_id - 1) if self.chain_id is not None else None
        identity = dict(self.identity)
        state_lane_id = identity.get("state_lane_id")
        if traj is not None:
            state_lane_id = agent_loop_lane_id(
                identity.get("experiment_name"),
                identity.get("sample"),
                identity.get("session"),
                traj,
            )
        _report_span(
            name="gateway_generation",
            start_time_ns=self.start_ns,
            identity=identity,
            attributes={
                "monitor.trace_source": "gateway",
                "state_lane_id": state_lane_id,
                "traj": traj,
                "status": self.status,
                "error": self.error,
                "finish_reason": self.finish_reason,
                "turn": self.turn,
                "type": self.type,
                "tools": self.tools,
                "content": self.content,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "chain_id": self.chain_id,
            },
        )


def start_generation_span(identity: dict[str, Any]) -> GenerationSpan:
    """Start one gateway-generation span."""
    return GenerationSpan(identity=dict(identity), start_ns=_start_span())


async def trace_sandbox_lifecycle(
    operation,
    *,
    sandbox: Any,
    lifecycle: str,
):
    """Observe one sandbox lifecycle operation and report its span."""
    provider = str(getattr(sandbox, "provider", "") or "")
    image = str(getattr(sandbox, "image", "") or "")
    runtime_id = _sandbox_runtime_id(sandbox)
    start_ns = _start_span()
    status, error = "success", None
    try:
        await operation
    except Exception as exc:
        status = "failure"
        error = f"{type(exc).__name__}:{exc}"
        raise
    finally:
        _report_span(
            name="agent_sandbox",
            start_time_ns=start_ns,
            attributes={
                "monitor.trace_source": "sandbox",
                "provider": provider,
                "image": image,
                "runtime_id": runtime_id,
                "lifecycle": lifecycle,
                "status": status,
                "error": error,
            },
        )


def _sandbox_runtime_id(sandbox: Any) -> str:
    for attr in ("_sandbox_id", "_container_name"):
        value = getattr(sandbox, attr, None)
        if value:
            return str(value)
    sandbox_obj = getattr(sandbox, "_sandbox", None)
    for attr in ("sandbox_id", "object_id", "id"):
        value = getattr(sandbox_obj, attr, None)
        if value:
            return str(value)
    return ""
