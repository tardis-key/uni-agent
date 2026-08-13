from __future__ import annotations

import asyncio
import os
import re
import signal
import uuid
from pathlib import Path

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox


class _LocalShell:
    """Long-lived local bash session for :meth:`LocalSandbox.open_shell`.

    ``provider: local`` runs on the host filesystem, so a native shell is a
    normal subprocess instead of the tmux/apt fallback used by remote-only
    providers.  Each command is delimited by unique markers written to stdout
    and stderr so output and the exit code can be captured without a PTY.
    """

    def __init__(
        self,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._cwd = cwd
        self._env = dict(env or {})
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return

        env = {**os.environ, "TERM": "dumb", "PS1": ""}
        env.update(self._env)
        self._proc = await asyncio.create_subprocess_exec(
            "bash",
            "--noprofile",
            "--norc",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=env,
            start_new_session=True,
        )

    def _process(self) -> asyncio.subprocess.Process:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            raise RuntimeError("local shell process is not running")
        return proc

    @staticmethod
    def _signal_process(proc: asyncio.subprocess.Process, sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

    async def _read_until(
        self,
        stream: asyncio.StreamReader,
        pattern: re.Pattern[str],
    ) -> tuple[str, int | None]:
        buf = ""
        try:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    return buf, None
                buf += chunk.decode("utf-8", "replace")
                match = pattern.search(buf)
                if match:
                    return buf[: match.start()], int(match.group(1))
        except asyncio.CancelledError:
            return buf, None

    async def _run_once(self, command: str, timeout: float | None) -> ExecResult:
        proc = self._process()
        token = uuid.uuid4().hex
        stdout_pattern = re.compile(
            rf"__UNI_AGENT_END_STDOUT_{token}_(-?\d+)__"
        )
        stderr_pattern = re.compile(
            rf"__UNI_AGENT_END_STDERR_{token}_(-?\d+)__"
        )
        script = (
            f"{command}\n"
            "__uni_rc=$?\n"
            'printf "\\n__UNI_AGENT_END_STDOUT_' + token + '_${__uni_rc}__\\n" >&1\n'
            'printf "\\n__UNI_AGENT_END_STDERR_' + token + '_${__uni_rc}__\\n" >&2\n'
        )

        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None
        try:
            proc.stdin.write(script.encode("utf-8"))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return ExecResult(
                exit_code=-1,
                stdout="",
                stderr="local shell process exited before the command could be sent",
            )

        stdout_task = asyncio.create_task(
            self._read_until(proc.stdout, stdout_pattern)
        )
        stderr_task = asyncio.create_task(
            self._read_until(proc.stderr, stderr_pattern)
        )
        pending = {stdout_task, stderr_task}
        done, pending = await asyncio.wait(pending, timeout=timeout)

        if pending:
            self._signal_process(proc, signal.SIGKILL)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._proc = None
            raise TimeoutError(f"local shell command timed out after {timeout:g}s")

        results = await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        stdout, stdout_rc = results[0] if not isinstance(results[0], BaseException) else ("", None)
        stderr, stderr_rc = results[1] if not isinstance(results[1], BaseException) else ("", None)
        exit_code = stdout_rc if stdout_rc is not None else stderr_rc
        if exit_code is None:
            exit_code = -1
        return ExecResult(
            exit_code=exit_code,
            stdout=_to_str(stdout),
            stderr=_to_str(stderr),
        )

    async def run(self, command: str, *, timeout: float | None = None) -> ExecResult:
        async with self._lock:
            await self.start()
            try:
                return await self._run_once(command, timeout)
            except (TimeoutError, asyncio.TimeoutError):
                proc = self._proc
                if proc is not None and proc.returncode is None:
                    self._signal_process(proc, signal.SIGKILL)
                    await proc.wait()
                self._proc = None
                raise

    async def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            self._signal_process(proc, signal.SIGKILL)
            await proc.wait()


@register_sandbox("local")
class LocalSandbox(Sandbox):
    """Runs commands on the host via ``asyncio`` subprocesses (no container).

    File operations use the host filesystem directly. Constructed with no args,
    so it uses the base :meth:`Sandbox.from_config` (which ignores the config
    fields).
    """

    supports_shell = True

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def read_file(self, path: str) -> bytes:
        """Read directly from the host filesystem without base64 transport."""
        return await asyncio.to_thread(Path(path).read_bytes)

    async def write_file(self, path: str, content: bytes | str) -> None:
        """Write directly to the host filesystem, creating parent directories."""
        data = content.encode("utf-8") if isinstance(content, str) else content
        target = Path(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(_write)

    async def open_shell(
        self,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _LocalShell:
        """Open a persistent local bash process; cwd/env persist across runs."""
        shell = _LocalShell(cwd=cwd, env=env)
        await shell.start()
        return shell

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env={**os.environ, **env} if env else None,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return ExecResult(
                exit_code=-1,
                stdout="",
                stderr=f"local exec timed out after {timeout}s",
            )
        return ExecResult(
            exit_code=proc.returncode or 0,
            stdout=_to_str(out),
            stderr=_to_str(err),
        )
