#!/usr/bin/env python3
"""Production-oriented CLI for autoDDPM operations."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".auto"
RUNS_FILE = STATE_DIR / "runs.json"
LOGS_DIR = ROOT / "logs"


@dataclass
class RunRecord:
    run_id: str
    mode: str
    status: str
    input_path: str
    output_dir: str
    job_id: str | None = None
    pid: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    created_at: str = ""
    command: str = ""


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dirs() -> None:
    STATE_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)


def load_state() -> dict[str, Any]:
    ensure_dirs()
    if not RUNS_FILE.exists():
        return {"runs": []}
    try:
        return json.loads(RUNS_FILE.read_text())
    except json.JSONDecodeError:
        return {"runs": []}


def save_state(state: dict[str, Any]) -> None:
    ensure_dirs()
    RUNS_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def add_run(record: RunRecord) -> None:
    state = load_state()
    state["runs"].append(asdict(record))
    save_state(state)


def update_run(run_id: str, **fields: Any) -> None:
    state = load_state()
    for run in state.get("runs", []):
        if run.get("run_id") == run_id:
            run.update(fields)
            break
    save_state(state)


def latest_run() -> dict[str, Any] | None:
    runs = load_state().get("runs", [])
    if not runs:
        return None
    return runs[-1]


def require_cmd(cmd: str) -> None:
    if shutil_which(cmd) is None:
        raise SystemExit(f"Required command not found in PATH: {cmd}")


def shutil_which(cmd: str) -> str | None:
    return subprocess.run(
        ["bash", "-lc", f"command -v {shlex.quote(cmd)}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or None


def run_shell(command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if check and proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        raise SystemExit(msg or f"Command failed: {command}")
    return proc


def build_inference_command(args: argparse.Namespace) -> str:
    cmd = [
        "python",
        "inference_clean.py",
        "--input",
        args.input,
        "--output",
        args.output,
        "--model_path",
        args.model_path,
        "--noise_recon",
        str(args.noise_recon),
        "--noise_inpaint",
        str(args.noise_inpaint),
        "--resample_steps",
        str(args.resample_steps),
        "--threshold",
        str(args.threshold),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    return " ".join(shlex.quote(part) for part in cmd)


def cmd_install(args: argparse.Namespace) -> None:
    venv = Path(args.venv).expanduser()
    if not venv.exists():
        print(f"Creating venv at {venv}")
        run_shell(f"python -m venv {shlex.quote(str(venv))}")
    activate = f"source {shlex.quote(str(venv / 'bin' / 'activate'))}"
    print("Installing base requirements...")
    run_shell(f"{activate} && pip install -r pip_requirements.txt")
    if args.with_torch:
        print("Installing PyTorch...")
        run_shell(
            f"{activate} && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        )
    if args.with_wandb:
        print("Installing wandb...")
        run_shell(f"{activate} && pip install wandb")
    print("Install complete.")


def cmd_start(args: argparse.Namespace) -> None:
    ensure_dirs()
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    cmd = build_inference_command(args)
    if args.slurm:
        require_cmd("sbatch")
        out_log = LOGS_DIR / f"auto_{run_id}_%j.out"
        err_log = LOGS_DIR / f"auto_{run_id}_%j.err"
        wrap = (
            f"source {shlex.quote(args.venv_activate)} && "
            f"cd {shlex.quote(str(ROOT))} && {cmd}"
        )
        sbatch_cmd = (
            "sbatch "
            f"--job-name={shlex.quote(args.job_name)} "
            f"--partition={shlex.quote(args.partition)} "
            f"--time={shlex.quote(args.time)} "
            f"--cpus-per-task={args.cpus} "
            f"--mem={shlex.quote(args.mem)} "
            f"--gres={shlex.quote(args.gres)} "
            f"--output={shlex.quote(str(out_log))} "
            f"--error={shlex.quote(str(err_log))} "
            f"--wrap={shlex.quote(wrap)}"
        )
        proc = run_shell(sbatch_cmd)
        out = proc.stdout.strip()
        if "Submitted batch job" not in out:
            raise SystemExit(f"Unexpected sbatch output: {out}")
        job_id = out.split()[-1]
        record = RunRecord(
            run_id=run_id,
            mode="slurm",
            status="submitted",
            input_path=str(input_path),
            output_dir=args.output,
            job_id=job_id,
            stdout_log=str(out_log).replace("%j", job_id),
            stderr_log=str(err_log).replace("%j", job_id),
            created_at=now_iso(),
            command=cmd,
        )
        add_run(record)
        print(f"Submitted job {job_id} (run_id={run_id})")
        print(f"stdout: {record.stdout_log}")
        print(f"stderr: {record.stderr_log}")
    else:
        out_log = LOGS_DIR / f"auto_{run_id}.out"
        err_log = LOGS_DIR / f"auto_{run_id}.err"
        full_cmd = (
            f"source {shlex.quote(args.venv_activate)} && "
            f"cd {shlex.quote(str(ROOT))} && {cmd}"
        )
        with out_log.open("w") as out_f, err_log.open("w") as err_f:
            proc = subprocess.Popen(
                ["bash", "-lc", full_cmd],
                stdout=out_f,
                stderr=err_f,
            )
        record = RunRecord(
            run_id=run_id,
            mode="local",
            status="running",
            input_path=str(input_path),
            output_dir=args.output,
            pid=proc.pid,
            stdout_log=str(out_log),
            stderr_log=str(err_log),
            created_at=now_iso(),
            command=cmd,
        )
        add_run(record)
        print(f"Started local run (pid={proc.pid}, run_id={run_id})")
        print(f"stdout: {out_log}")
        print(f"stderr: {err_log}")


def cmd_stop(args: argparse.Namespace) -> None:
    run = None
    if args.run_id:
        runs = load_state().get("runs", [])
        run = next((r for r in runs if r.get("run_id") == args.run_id), None)
    else:
        run = latest_run()
    if run is None:
        raise SystemExit("No run found to stop.")
    if run.get("mode") == "slurm":
        job_id = run.get("job_id")
        if not job_id:
            raise SystemExit("Run has no Slurm job ID.")
        require_cmd("scancel")
        run_shell(f"scancel {shlex.quote(str(job_id))}")
        update_run(run["run_id"], status="stopped")
        print(f"Stopped Slurm job {job_id}")
    else:
        pid = run.get("pid")
        if not pid:
            raise SystemExit("Run has no local PID.")
        run_shell(f"kill {int(pid)}", check=False)
        update_run(run["run_id"], status="stopped")
        print(f"Stopped local process {pid}")


def find_log_path(path: str | None) -> str | None:
    if not path:
        return None
    if "%" not in path:
        return path if Path(path).exists() else None
    matches = glob.glob(path.replace("%j", "*"))
    return matches[-1] if matches else None


def cmd_logs(args: argparse.Namespace) -> None:
    run = None
    runs = load_state().get("runs", [])
    if args.run_id:
        run = next((r for r in runs if r.get("run_id") == args.run_id), None)
    elif args.job_id:
        run = next((r for r in runs if r.get("job_id") == args.job_id), None)
    else:
        run = latest_run()
    if run is None:
        raise SystemExit("No run found.")
    log_path = find_log_path(run.get("stdout_log") if args.stream != "stderr" else run.get("stderr_log"))
    if not log_path:
        raise SystemExit("Log file not found.")
    if args.follow:
        os.execvp("tail", ["tail", "-n", str(args.lines), "-f", log_path])
    else:
        out = run_shell(f"tail -n {args.lines} {shlex.quote(log_path)}", check=True)
        print(out.stdout, end="")


def cmd_checks(args: argparse.Namespace) -> None:
    checks: list[tuple[str, bool, str]] = []
    py_ver = tuple(int(x) for x in sys.version.split()[0].split(".")[:2])
    checks.append(("python", py_ver >= (3, 8), sys.version.split()[0]))
    checks.append(("latest_model.pt", (ROOT / "latest_model.pt").exists(), "required for inference"))
    checks.append(("inference_clean.py", (ROOT / "inference_clean.py").exists(), "entrypoint"))
    gpu = run_shell("python - <<'PY'\nimport torch\nprint(torch.cuda.is_available())\nPY", check=False).stdout.strip()
    checks.append(("cuda_available", gpu == "True", gpu or "unknown"))
    if args.input:
        checks.append(("input_exists", Path(args.input).expanduser().exists(), args.input))
    if args.output:
        out = Path(args.output).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        checks.append(("output_writable", os.access(out, os.W_OK), str(out)))
    if shutil_which("sbatch"):
        checks.append(("slurm_sbatch", True, "available"))
    else:
        checks.append(("slurm_sbatch", False, "not found"))

    failed = False
    for name, ok, detail in checks:
        mark = "OK" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        if not ok and name in {"latest_model.pt", "inference_clean.py", "input_exists"}:
            failed = True
    if failed:
        raise SystemExit(1)


def cmd_status(_: argparse.Namespace) -> None:
    run = latest_run()
    if not run:
        print("No runs found.")
        return
    print(json.dumps(run, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="autoDDPM production CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_install = sub.add_parser("install", help="Set up environment and dependencies")
    p_install.add_argument("--venv", default="~/.venvs/autoddpm", help="Virtualenv path")
    p_install.add_argument("--with-torch", action="store_true", help="Install torch+torchvision")
    p_install.add_argument("--with-wandb", action="store_true", help="Install wandb")
    p_install.set_defaults(func=cmd_install)

    p_start = sub.add_parser("start", help="Start an inference run")
    p_start.add_argument("--input", required=True, help="Input NIfTI file")
    p_start.add_argument("--output", required=True, help="Output directory")
    p_start.add_argument("--model-path", default="./latest_model.pt")
    p_start.add_argument("--noise-recon", type=int, default=200)
    p_start.add_argument("--noise-inpaint", type=int, default=50)
    p_start.add_argument("--resample-steps", type=int, default=5)
    p_start.add_argument("--threshold", type=float, default=-1)
    p_start.add_argument("--batch-size", type=int, default=8)
    p_start.add_argument("--device", default="cuda")
    p_start.add_argument("--slurm", action="store_true", default=True, help="Run via sbatch (default on)")
    p_start.add_argument("--local", dest="slurm", action="store_false", help="Run as local background process")
    p_start.add_argument("--venv-activate", default="~/.venvs/autoddpm/bin/activate")
    p_start.add_argument("--job-name", default="autoDDPM_prod")
    p_start.add_argument("--partition", default="general")
    p_start.add_argument("--time", default="02:00:00")
    p_start.add_argument("--cpus", type=int, default=8)
    p_start.add_argument("--mem", default="32G")
    p_start.add_argument("--gres", default="gpu:l40s.12g:1")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop a run")
    p_stop.add_argument("--run-id", help="Run ID to stop (defaults to latest)")
    p_stop.set_defaults(func=cmd_stop)

    p_logs = sub.add_parser("logs", help="Show logs for a run")
    p_logs.add_argument("--run-id", help="Run ID")
    p_logs.add_argument("--job-id", help="Slurm job ID")
    p_logs.add_argument("--stream", choices=["stdout", "stderr"], default="stdout")
    p_logs.add_argument("--lines", type=int, default=80)
    p_logs.add_argument("--follow", action="store_true")
    p_logs.set_defaults(func=cmd_logs)

    p_checks = sub.add_parser("checks", help="Run readiness checks")
    p_checks.add_argument("--input", help="Optional input file to validate")
    p_checks.add_argument("--output", help="Optional output directory to validate")
    p_checks.set_defaults(func=cmd_checks)

    p_status = sub.add_parser("status", help="Show latest run metadata")
    p_status.set_defaults(func=cmd_status)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
