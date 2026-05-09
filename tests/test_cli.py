import os
import subprocess
import sys


def test_print_config_runs(tmp_path):
    env = os.environ.copy()
    env["AGENT_WORKSPACE"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "agent_learning", "--print-config"],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    assert "workspace" in proc.stdout
    assert "azure_configured" in proc.stdout
