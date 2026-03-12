# VM Army Deployment — How It Was Done

**Date:** 2026-03-12
**Budget:** $200 / 10 days
**Actual estimated cost:** ~$128

---

## What Was Deployed

| Resource | Details |
|---|---|
| **10× Azure VMs** | `Standard_D2s_v3` (2 vCPU, 8GB RAM) in `eastus2` |
| **10× OpenAI deployments** | `gpt-5-nano` slots on `noissues-52` (250K TPM each) |
| **1× Storage container** | `noissuesresults/results` — receives logs + Dockerfiles from VMs |
| **Resource group** | `noissues` |

Each VM processes ~28–29 repos (282 total split evenly), using 2 parallel workers.
Estimated runtime: ~36h per VM, then auto-deallocate.

---

## Architecture

```
Your machine
    │
    ├─ Splits 282 repos into 10 shards
    ├─ Uploads setup scripts to blob storage
    └─ az vm create ×10 (parallel, isolated az config dirs)
           │
           ▼
    noissues-vm01 ──► gpt-5-nano        (250K TPM)
    noissues-vm02 ──► gpt-5-nano-slot2  (250K TPM)
    ...
    noissues-vm10 ──► gpt-5-nano-slot10 (250K TPM)
           │
           ▼ (on completion)
    noissuesresults blob storage
    └── noissues-vm01/
    │   ├── parallel_empirical_results/
    │   │   ├── results_*.jsonl
    │   │   ├── artifacts/  (Dockerfiles)
    │   │   ├── agent_transcripts/
    │   │   └── summary_*.txt
    │   ├── run.log
    │   └── setup.log
    └── noissues-vm02/ ...
```

---

## Step-by-Step Breakdown

### 1. Surveyed existing Azure OpenAI resources
```bash
az cognitiveservices account list
az cognitiveservices account deployment list --name noissues-52 --resource-group noissues
az cognitiveservices usage list --location eastus2
```
Found `gpt-5-nano` at 250K TPM used out of **150M TPM** quota — huge headroom.

### 2. Created 9 extra gpt-5-nano deployment slots
```bash
for i in $(seq 2 10); do
  az cognitiveservices account deployment create \
    --resource-group noissues \
    --name noissues-52 \
    --deployment-name "gpt-5-nano-slot${i}" \
    --model-name "gpt-5-nano" \
    --model-version "2025-08-07" \
    --model-format OpenAI \
    --sku-capacity 250 \
    --sku-name GlobalStandard
done
```
Each slot gets its own 250K TPM budget — no rate limit collisions between VMs.

### 3. Created blob storage for results
```bash
az storage account create --name noissuesresults --resource-group noissues --location eastus
az storage container create --name results --account-name noissuesresults
```
Generated a SAS token valid until 2026-03-28 with `racwdl` permissions.

### 4. Split repos into 10 shards
282 repos from `sampled_repos_urls.txt` → 10 files of ~28 repos each in `/tmp/noissues-deploy/repo-shards/`.

### 5. Generated SSH key
```bash
ssh-keygen -t ed25519 -f ~/.ssh/noissues-vms
```

### 6. Generated per-VM setup scripts + uploaded to blob
Each `setup-vm{N}.sh` script:
1. Moves Docker data-root to `/mnt` (temp disk) for faster I/O
2. Installs Python 3.12 + git
3. Clones `https://github.com/Vibe-Checkers/NoIssues.git`
4. Creates virtualenv and installs `requirements.txt`
5. Writes `.env` with the VM's dedicated gpt-5-nano slot
6. Writes the repos shard to `/home/azureuser/repos.txt`
7. Runs `python src/parallel_empirical_test.py repos.txt --workers 2`
8. Uploads all results + logs to blob via `azcopy`
9. Deallocates itself: `az vm deallocate ...`

Scripts uploaded to `noissuesresults/results/scripts/setup-vm{01-10}.sh`.

### 7. Generated tiny cloud-init per VM
Cloud-init only does one thing — download and run the setup script:
```yaml
#cloud-config
runcmd:
  - curl -sL "<blob_url_with_sas>" -o /home/azureuser/setup.sh
  - chmod +x /home/azureuser/setup.sh
  - bash /home/azureuser/setup.sh
```
This avoids cloud-init YAML size limits and parsing issues.

### 8. Launched 10 VMs in parallel
```bash
# Run with isolated az config dirs to avoid HTTP session sharing bug in az-cli 2.84
for i in $(seq 1 10); do
  AZURE_CONFIG_DIR="/tmp/azcfg/vm${i}" az vm create \
    --resource-group noissues \
    --name "noissues-vm$(printf '%02d' $i)" \
    --location eastus2 \
    --size Standard_D2s_v3 \
    --image Ubuntu2204 \
    --admin-username azureuser \
    --ssh-key-values "$PUB_KEY" \
    --custom-data "@cloud-init-vm$(printf '%02d' $i).yaml" \
    --public-ip-sku Standard &
done
wait
```

> **Note:** `eastus2` was chosen because `eastus` had no D2-series capacity available.
> `Standard_D2s_v3` was the first size that passed `--validate` in eastus2.
> `--no-wait` causes a bug in az-cli 2.84 ("content already consumed"), so background `&` was used instead with isolated `AZURE_CONFIG_DIR`.

---

## VM IPs

| VM | IP | Deployment Slot |
|---|---|---|
| noissues-vm01 | 20.7.162.209 | gpt-5-nano |
| noissues-vm02 | 104.46.126.89 | gpt-5-nano-slot2 |
| noissues-vm03 | 52.179.215.176 | gpt-5-nano-slot3 |
| noissues-vm04 | 20.10.32.126 | gpt-5-nano-slot4 |
| noissues-vm05 | 20.1.168.131 | gpt-5-nano-slot5 |
| noissues-vm06 | 20.110.202.1 | gpt-5-nano-slot6 |
| noissues-vm07 | 20.110.170.77 | gpt-5-nano-slot7 |
| noissues-vm08 | 20.12.25.182 | gpt-5-nano-slot8 |
| noissues-vm09 | 20.114.237.41 | gpt-5-nano-slot9 |
| noissues-vm10 | 20.110.75.114 | gpt-5-nano-slot10 |

---

## Monitoring & Operations

**Monitor all VMs:**
```bash
python monitor_vms.py           # one-shot
python monitor_vms.py --watch   # refresh every 30s
```

**SSH into a VM:**
```bash
ssh -i ~/.ssh/noissues-vms azureuser@20.7.162.209
tail -f /home/azureuser/setup.log   # bootstrap progress
tail -f /home/azureuser/run.log     # test run output
```

**Download all results when done:**
```bash
azcopy copy \
  'https://noissuesresults.blob.core.windows.net/results/?se=2026-03-28&sp=racwdl&sv=2026-02-06&sr=c&sig=ohGy6/H9kCpA2yZ93xjYvch6MdWtaYrgHwZX6ZHQuQw%3D' \
  ./results --recursive
```

**Emergency: stop all VMs immediately:**
```bash
for i in $(seq 1 10); do
  az vm deallocate --resource-group noissues --name "noissues-vm$(printf '%02d' $i)" --no-wait
done
```

**Delete all VMs when done:**
```bash
for i in $(seq 1 10); do
  az vm delete --resource-group noissues --name "noissues-vm$(printf '%02d' $i)" --yes --no-wait
done
```

---

## Cost Breakdown

| Item | Est. Cost |
|---|---|
| 10× D2s_v3 × ~36h × $0.094/hr | ~$34 |
| gpt-5-nano API (282 repos × ~$0.33) | ~$93 |
| Blob storage (negligible) | ~$0 |
| **Total** | **~$127** |

gpt-5-nano pricing: $0.15/1M input tokens, $0.60/1M output tokens.

---

## Files Created

| File | Purpose |
|---|---|
| `monitor_vms.py` | Live monitoring of all 10 VMs |
| `VM_DEPLOYMENT.md` | This document |
| `~/.ssh/noissues-vms` | SSH private key |
| `~/.ssh/noissues-vms.pub` | SSH public key |
| `/tmp/noissues-deploy/` | Deployment artifacts (scripts, cloud-init, shards) |
