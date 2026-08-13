from __future__ import annotations

import asyncio

import pytest

from uni_agent.sandbox.local import LocalSandbox


def test_local_sandbox_advertises_native_shell():
    sandbox = LocalSandbox()

    assert sandbox.supports_shell is True
    assert "open_shell" in LocalSandbox.__dict__


def test_open_shell_captures_stdout_and_exit_code(tmp_path):
    async def scenario():
        sandbox = LocalSandbox()
        shell = await sandbox.open_shell(cwd=str(tmp_path))
        try:
            cwd_result = await shell.run("pwd")
            assert cwd_result.exit_code == 0
            assert cwd_result.stdout.strip() == str(tmp_path)
            assert cwd_result.stderr.strip() == ""

            ok_result = await shell.run("printf 'ok'")
            assert ok_result.exit_code == 0
            assert ok_result.stdout.strip() == "ok"

            failed_result = await shell.run("false")
            assert failed_result.exit_code == 1
            assert failed_result.stdout.strip() == ""
            assert failed_result.stderr.strip() == ""
        finally:
            await shell.close()

    asyncio.run(scenario())


def test_open_shell_persists_cwd_and_environment(tmp_path):
    async def scenario():
        sandbox = LocalSandbox()
        shell = await sandbox.open_shell(cwd=str(tmp_path), env={"FOO": "bar"})
        try:
            env_result = await shell.run('printf "%s" "$FOO"')
            assert env_result.exit_code == 0
            assert env_result.stdout.strip() == "bar"

            export_result = await shell.run('export ANSWER=42; printf "%s" "$ANSWER"')
            assert export_result.exit_code == 0
            assert export_result.stdout.strip() == "42"

            cd_result = await shell.run("cd /tmp && pwd")
            assert cd_result.exit_code == 0
            assert cd_result.stdout.strip() == "/tmp"

            persisted_cwd_result = await shell.run("pwd")
            assert persisted_cwd_result.exit_code == 0
            assert persisted_cwd_result.stdout.strip() == "/tmp"
        finally:
            await shell.close()

    asyncio.run(scenario())


def test_open_shell_times_out_and_cleanly_stops_subprocess():
    async def scenario():
        sandbox = LocalSandbox()
        shell = await sandbox.open_shell()
        try:
            with pytest.raises(TimeoutError, match="timed out"):
                await shell.run("sleep 5", timeout=0.2)
        finally:
            await shell.close()
            await shell.close()

    asyncio.run(scenario())
