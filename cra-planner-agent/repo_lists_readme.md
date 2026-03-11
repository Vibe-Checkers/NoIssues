# Repository List Files

These text files track the GitHub repositories used across multiple rounds of empirical testing of our Dockerfile-generation agent. Each round ran the agent on a set of repos across distributed machines (m0, m4, m5, murgohu).

## Source / Input Lists

| File | Lines | Description |
|------|-------|-------------|
| `sampled_repos_urls.txt` | 282 | **Original sample** — 282 GitHub repo URLs randomly sampled for the empirical evaluation. This is the starting point for all rounds. |
| `library_links.txt` | 49 | **Library repos** — subset of repos classified as libraries. Combined with `failed_repos.txt` for round 2 retries. |
| `test_repos.txt` | 2 | **Test set** — 2 repos used for quick smoke-testing the pipeline before full runs. |
| `test_2repos.txt` | 2 | **Test set (alt)** — another 2-repo test file for pipeline debugging. |

## Round 1 Results (282 repos across m0, m4, m5)

| File | Lines | Description |
|------|-------|-------------|
| `successful_repos.txt` | 57 | Repos that successfully generated a working Dockerfile in round 1. |
| `failed_repos.txt` | 173 | Repos that failed in round 1 (build errors, timeouts, rate limits, etc.). |

## Round 2 Results (failed + library repos across m0, m4, m5)

| File | Lines | Description |
|------|-------|-------------|
| `retry_repos.txt` | 219 | Input for round 2 — `failed_repos.txt` (173) + `library_links.txt` (49) minus duplicates. |
| `successful_repos_r2.txt` | 8 | New successes from round 2. |
| `failed_repos_r2.txt` | 11 | Repos that still failed after round 2 (subset that was retried with focused attention). |

## Round 3 Results (retry repos across m0, m4, m5, murgohu)

| File | Lines | Description |
|------|-------|-------------|
| `successful_repos_r3.txt` | 32 | New successes from round 3. Murgohu (4th machine) handled 3x the load. |

## Aggregate

| File | Lines | Description |
|------|-------|-------------|
| `all_successful_repos.txt` | 96 | **Combined successes** across all 3 rounds (57 + 8 + 32 = 97, minus 1 duplicate = 96). |

## Round 4 (current — improved agent)

| File | Lines | Description |
|------|-------|-------------|
| `failed_repos_r4.txt` | 90 | Repos that failed in round 3 (excluding 44 skipped/documentation-only). These are being retried with the improved agent (new prompt rules, expanded failure classifier, enhanced metaprompt). |
| `failed_repos_r4_part_aa` | 30 | Split for m0. |
| `failed_repos_r4_part_ab` | 30 | Split for m4. |
| `failed_repos_r4_part_ac` | 30 | Split for m5. |

## Pipeline Summary

```
Round 1: 282 repos -> 57 success, 173 fail, 52 skip
Round 2: 219 repos -> 8 new success, 11 tracked fails
Round 3: 211 repos -> 32 new success, 90 fail, 44 skip  (added murgohu)
Round 4: 90 repos  -> in progress (improved agent v2)
Total successes so far: 96/282 (34%)
```
