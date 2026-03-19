#!/usr/bin/env python3
"""Manage a tagged cron job for periodic Docker cleanup.

Commands:
    install   Install or update a tagged cron entry (idempotent)
    remove    Remove tagged cron entry if present (idempotent)
    status    Show whether tagged cron entry is installed

Default schedule is every 30 minutes and runs:
    docker system prune --force --filter "until=1h"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

TAG = "NOISSUES_DOCKER_PRUNE"
DEFAULT_SCHEDULE = "*/30 * * * *"
DEFAULT_FILTER = "until=1h"
LOG_RELATIVE = "logs/docker-prune-cron.log"


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def run_checked(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
    )


def check_cron_available() -> None:
    if shutil.which("crontab"):
        return
    # Fallback path for system-wide mode when user crontab binary is unavailable.
    if not Path("/etc/crontab").exists():
        eprint("ERROR: cron is not available (missing `crontab` command and `/etc/crontab`).")
        sys.exit(2)


def check_docker_available() -> None:
    if shutil.which("docker") is None:
        eprint("ERROR: docker command not found in PATH.")
        sys.exit(2)


def check_docker_daemon() -> None:
    proc = run_checked(["docker", "info"])
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        eprint(f"ERROR: Docker daemon is not reachable. {details}")
        sys.exit(3)


def workspace_root() -> Path:
    # Script is in scripts/, so parent of parent is project root.
    return Path(__file__).resolve().parent.parent


def ensure_log_dir(root: Path) -> Path:
    log_path = root / LOG_RELATIVE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def cron_command(root: Path, log_path: Path, prune_filter: str) -> str:
    # Use /bin/zsh -lc to preserve shell behavior on macOS and include timestamps.
    root_s = str(root)
    log_s = str(log_path)
    ts_fmt = "+%Y-%m-%dT%H:%M:%S%z"
    return (
        f"cd {shell_quote(root_s)} && /bin/zsh -lc "
        + shell_quote(
            f"printf \"[%s] docker-prune start\\n\" \"$(date '{ts_fmt}')\" >> {shell_quote(log_s)}; "
            + f"yes | docker system prune --force --filter {shell_quote(prune_filter)} >> {shell_quote(log_s)} 2>&1; "
            + f"rc=$?; printf \"[%s] docker-prune exit=%s\\n\" \"$(date '{ts_fmt}')\" \"$rc\" >> {shell_quote(log_s)}"
        )
    )


def shell_quote(value: str) -> str:
    # Minimal single-quote escaping for shell-safe literal strings.
    return "'" + value.replace("'", "'\"'\"'") + "'"


def user_crontab_read() -> tuple[list[str], bool]:
    proc = run_checked(["crontab", "-l"])
    if proc.returncode == 0:
        lines = [ln for ln in proc.stdout.splitlines()]
        return lines, True

    text = (proc.stderr or proc.stdout or "").lower()
    if "no crontab for" in text or "no crontab" in text:
        return [], True

    return [], False


def user_crontab_write(lines: list[str]) -> bool:
    payload = "\n".join(lines).rstrip() + "\n"
    proc = run_checked(["crontab", "-"], input_text=payload)
    return proc.returncode == 0


def system_crontab_path() -> Path:
    return Path("/etc/crontab")


def system_crontab_read() -> tuple[list[str], bool]:
    path = system_crontab_path()
    if not path.exists():
        return [], False
    try:
        return path.read_text().splitlines(), True
    except Exception:
        return [], False


def system_crontab_write(lines: list[str]) -> bool:
    path = system_crontab_path()
    payload = "\n".join(lines).rstrip() + "\n"
    try:
        path.write_text(payload)
        return True
    except PermissionError:
        eprint("ERROR: cannot write /etc/crontab. Re-run with sudo or use user crontab.")
        return False
    except Exception as exc:
        eprint(f"ERROR: failed to update /etc/crontab: {exc}")
        return False


def build_user_entry(schedule: str, command: str) -> str:
    return f"{schedule} {command} # {TAG}"


def build_system_entry(schedule: str, command: str) -> str:
    user = os.environ.get("USER") or "root"
    return f"{schedule} {user} {command} # {TAG}"


def remove_tagged(lines: list[str]) -> list[str]:
    return [ln for ln in lines if TAG not in ln]


def install_entry(schedule: str, prune_filter: str) -> int:
    root = workspace_root()
    log_path = ensure_log_dir(root)
    command = cron_command(root, log_path, prune_filter)

    # Prefer user crontab.
    if shutil.which("crontab"):
        lines, ok = user_crontab_read()
        if ok:
            lines = remove_tagged(lines)
            lines.append(build_user_entry(schedule, command))
            if not user_crontab_write(lines):
                eprint("ERROR: failed to install cron entry into user crontab.")
                return 4
            print("Installed Docker prune cron entry in user crontab.")
            return 0

    # Fallback: /etc/crontab
    lines, ok = system_crontab_read()
    if not ok:
        eprint("ERROR: user crontab unavailable and /etc/crontab is not readable.")
        return 4
    lines = remove_tagged(lines)
    lines.append(build_system_entry(schedule, command))
    if not system_crontab_write(lines):
        return 4
    print("Installed Docker prune cron entry in /etc/crontab.")
    return 0


def remove_entry() -> int:
    removed_any = False

    if shutil.which("crontab"):
        lines, ok = user_crontab_read()
        if ok:
            new_lines = remove_tagged(lines)
            if new_lines != lines:
                if not user_crontab_write(new_lines):
                    eprint("ERROR: failed to remove cron entry from user crontab.")
                    return 4
                removed_any = True

    lines, ok = system_crontab_read()
    if ok:
        new_lines = remove_tagged(lines)
        if new_lines != lines:
            if not system_crontab_write(new_lines):
                return 4
            removed_any = True

    if removed_any:
        print("Removed Docker prune cron entry.")
    else:
        print("Docker prune cron entry not present. Nothing to remove.")
    return 0


def status_entry() -> int:
    found = []

    if shutil.which("crontab"):
        lines, ok = user_crontab_read()
        if ok:
            for ln in lines:
                if TAG in ln:
                    found.append(("user", ln))

    lines, ok = system_crontab_read()
    if ok:
        for ln in lines:
            if TAG in ln:
                found.append(("system", ln))

    if not found:
        print("Status: NOT INSTALLED")
        return 0

    print("Status: INSTALLED")
    for scope, line in found:
        print(f"- {scope}: {line}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage cron scheduling for periodic docker system prune.",
    )
    parser.add_argument(
        "command",
        choices=["install", "remove", "status"],
        help="Operation to run",
    )
    parser.add_argument(
        "--schedule",
        default=DEFAULT_SCHEDULE,
        help=f"Cron schedule expression (default: '{DEFAULT_SCHEDULE}')",
    )
    parser.add_argument(
        "--filter",
        default=DEFAULT_FILTER,
        help=f"Docker prune --filter value (default: '{DEFAULT_FILTER}')",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    check_cron_available()
    check_docker_available()
    check_docker_daemon()

    if args.command == "install":
        return install_entry(args.schedule, args.filter)
    if args.command == "remove":
        return remove_entry()
    if args.command == "status":
        return status_entry()

    eprint(f"ERROR: unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
