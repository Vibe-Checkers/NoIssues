#!/usr/bin/env bash
set -euo pipefail

RG="${RG:-noissues}"
VM_PREFIX="${VM_PREFIX:-noissues-vm}"
SSH_USER="${SSH_USER:-azureuser}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/noissues-vms}"
RUN_STAMP_FILTER="${RUN_STAMP_FILTER:-}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }
}

require_cmd az
require_cmd ssh
require_cmd python3

[[ -f "$SSH_KEY" ]] || { echo "SSH key not found: $SSH_KEY" >&2; exit 1; }

VM_LIST_FILE="$(mktemp)"
az vm list --resource-group "$RG" --show-details --output json \
| python3 -c 'import json,sys
vms=json.load(sys.stdin)
rows=[]
for vm in vms:
    name=vm.get("name","")
    ip=vm.get("publicIps","")
    if name.startswith("'"$VM_PREFIX"'") and ip:
        rows.append((name, ip))
for name, ip in sorted(rows):
    print(f"{name} {ip}")' > "$VM_LIST_FILE"

if [[ ! -s "$VM_LIST_FILE" ]]; then
  echo "No VMs found with prefix '$VM_PREFIX' in resource group '$RG'." >&2
  rm -f "$VM_LIST_FILE"
  exit 1
fi

printf "%-16s %-16s %-10s %-8s %-8s %-8s %s\n" "VM" "IP" "STATE" "PYVENV" "PROCESS" "RUNID" "LOG_TAIL"
printf '%s\n' "---------------------------------------------------------------------------------------------------------------"

ok=0
bad=0

while IFS= read -r line; do
  vm_name="${line%% *}"
  vm_ip="${line##* }"

  out="$(ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$SSH_USER@$vm_ip" '
set +e
STATE="ok"
if [ -x /home/azureuser/NoIssues/cra-planner-agent/.venv/bin/python ] || [ -x /home/azureuser/NoIssues/cra-planner-agent/cra-planner-agent/.venv/bin/python ]; then
  PYVENV="yes"
else
  PYVENV="no"
fi

if pgrep -f "src/parallel_empirical_test.py" >/dev/null 2>&1; then
  PROCESS="yes"
else
  PROCESS="no"
fi

RUNID=$(cat /home/azureuser/vmtest-run-id.txt 2>/dev/null || echo none)
TAIL=$(tail -1 /home/azureuser/vmtest-run.log 2>/dev/null | tr "\t" " " | cut -c1-60)

echo "$STATE|$PYVENV|$PROCESS|$RUNID|$TAIL"
' 2>/dev/null || echo "ssh_fail|no|no|none|-")"

state="${out%%|*}"
rest="${out#*|}"
pyvenv="${rest%%|*}"; rest="${rest#*|}"
process="${rest%%|*}"; rest="${rest#*|}"
runid="${rest%%|*}"; tail="${rest#*|}"

printf "%-16s %-16s %-10s %-8s %-8s %-8s %s\n" "$vm_name" "$vm_ip" "$state" "$pyvenv" "$process" "$runid" "$tail"

  stamp_ok="yes"
  if [[ -n "$RUN_STAMP_FILTER" && "$runid" != "$RUN_STAMP_FILTER" ]]; then
    stamp_ok="no"
  fi

  if [[ "$state" == "ok" && "$pyvenv" == "yes" && "$process" == "yes" && "$stamp_ok" == "yes" ]]; then
    ((ok+=1))
  else
    ((bad+=1))
  fi
done < "$VM_LIST_FILE"

rm -f "$VM_LIST_FILE"

echo
echo "Healthy VMs: $ok"
echo "Unhealthy VMs: $bad"
if [[ -n "$RUN_STAMP_FILTER" ]]; then
  echo "Run stamp filter: $RUN_STAMP_FILTER"
fi

if [[ $bad -gt 0 ]]; then
  exit 2
fi
