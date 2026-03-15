# VM Reset + Fresh Run (Fool-Proof)

This runbook uses:
- VM inventory: [`scripts/vm_inventory.tsv`](NoIssues/cra-planner-agent/scripts/vm_inventory.tsv)
- Remote hard cleaner: [`scripts/vm_absolute_clean_remote.sh`](NoIssues/cra-planner-agent/scripts/vm_absolute_clean_remote.sh)
- Orchestrator (distribute clean + run): [`scripts/vm_distribute_clean_and_run.sh`](NoIssues/cra-planner-agent/scripts/vm_distribute_clean_and_run.sh)
- Health checker: [`scripts/vm_check_run_health.sh`](NoIssues/cra-planner-agent/scripts/vm_check_run_health.sh)
- Existing monitor: [`monitor_vms.py`](NoIssues/cra-planner-agent/monitor_vms.py)

## What this solves

The orchestrator performs a deterministic reset on **all 10 VMs** and relaunches from scratch:
1. Distributes and executes absolute cleaner (remote)
2. **Absolute clean**: hard wipe everything under `/home/azureuser` except `.ssh`
3. Kills test/build processes and removes Docker containers/images/volumes/cache
4. Reinstalls base tooling (`python3`, `python3-venv`, `python3-pip`, `git`, `curl`)
5. Reclones repo and creates fresh virtualenv
6. Overlays local patched files:
   - [`src/parallel_empirical_test.py`](NoIssues/cra-planner-agent/src/parallel_empirical_test.py)
   - [`src/agent/workflow.py`](NoIssues/cra-planner-agent/src/agent/workflow.py)
   - [`src/agent/tools.py`](NoIssues/cra-planner-agent/src/agent/tools.py)
   - [`.env`](NoIssues/cra-planner-agent/.env)
7. Starts fresh run with [`--workers 1`](NoIssues/cra-planner-agent/src/parallel_empirical_test.py:1)

## Prerequisites

On your local machine:
- Azure CLI logged in and authorized for resource group `noissues`
- SSH key exists at `/Users/afikbae/.ssh/noissues-vms`
- Project exists at `/Users/afikbae/Clones/capstone/NoIssues/cra-planner-agent`

## One-command absolute clean + run

From workspace root:

```bash
cd /Users/afikbae/Clones/capstone/NoIssues/cra-planner-agent
bash scripts/vm_distribute_clean_and_run.sh
```

Optional flags:

```bash
WORKERS=1 REPO_BRANCH=vm-testing PER_REPO_TIMEOUT_SECONDS=7200 \
bash scripts/vm_distribute_clean_and_run.sh
```

## Verify health immediately

```bash
cd /Users/afikbae/Clones/capstone/NoIssues/cra-planner-agent
bash scripts/vm_check_run_health.sh
```

Expected healthy state per VM:
- `PYVENV=yes`
- `PROCESS=yes`

## Live monitoring

Use existing monitor:

```bash
cd /Users/afikbae/Clones/capstone/NoIssues/cra-planner-agent
python3 monitor_vms.py --watch
```

## Inventory format

File: [`scripts/vm_inventory.tsv`](NoIssues/cra-planner-agent/scripts/vm_inventory.tsv)

Columns:
1. vm_name
2. ip
3. ssh_user
4. slot (label)
5. remote_repos_file

Use tab-separated values. Lines starting with `#` are comments.

## If any VM fails to start

1. Re-run full clean+run orchestrator (idempotent):

```bash
bash scripts/vm_distribute_clean_and_run.sh
```

2. Inspect failing VM directly:

```bash
ssh -i /Users/afikbae/.ssh/noissues-vms azureuser@<VM_IP>
tail -n 200 /home/azureuser/vmtest-run.log
```

3. Verify virtualenv + deps:

```bash
ssh -i /Users/afikbae/.ssh/noissues-vms azureuser@<VM_IP> \
  '/home/azureuser/NoIssues/cra-planner-agent/.venv/bin/python -c "import dotenv, langchain_openai; print(\"ok\")"'
```

## Recovery policy (safe default)

If uncertain, always do full reset + re-run with:
- [`scripts/vm_distribute_clean_and_run.sh`](NoIssues/cra-planner-agent/scripts/vm_distribute_clean_and_run.sh)
- then [`scripts/vm_check_run_health.sh`](NoIssues/cra-planner-agent/scripts/vm_check_run_health.sh)

This is safer than patching ad-hoc state on partially configured VMs.
