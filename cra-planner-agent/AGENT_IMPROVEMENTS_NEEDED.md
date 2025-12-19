# Agent Improvements Needed - Based on dubbo Analysis

## 🔍 What We Learned from dubbo

**dubbo failed across 10+ iterations** with a clear pattern:
- **Iterations 0-4**: FILE_COPY_MISSING (missing `src` directory)
- **Iterations 5-8**: PLATFORM_INCOMPATIBLE (alpine doesn't support ARM64)
- **Iterations 3, 6**: IMAGE_PULL (openjdk:17-jdk-slim doesn't exist)

**Agent behavior**:
- Flips between `eclipse-temurin:17_35-jdk-alpine` ← → `openjdk:17-jdk-slim`
- **NEVER fixes the COPY command** (real root cause!)
- Gets stuck changing images instead of fixing the actual problem

---

## 🚨 Core Issues

### Issue #1: Agent Not Reading Error Messages Carefully

**Evidence**:
```
Warning: Missing COPY sources: src
→ Agent gets: "FILE_COPY_MISSING" error
→ Prompt says: "DO NOT CHANGE BASE IMAGE! Fix COPY path"
→ Agent does: Changes base image anyway (ignores instruction!)
```

**Why**: Agent sees multiple signals (image error + copy error) and focuses on wrong one.

### Issue #2: PLATFORM_INCOMPATIBLE Should Trigger Immediate Image Change

**Current behavior**:
```
Error: eclipse-temurin:17_35-jdk-alpine doesn't support ARM64
→ Agent gets: DEPENDENCY_BUILD_TOOLS prompt (WRONG!)
→ Agent tries: Installing gcc, make (irrelevant!)
→ Should do: Use DockerImageSearch to find ARM64-compatible image
```

**Missing**: Specific handler for PLATFORM_INCOMPATIBLE

### Issue #3: No Validation of Image Choice

**What happens**:
```
Agent picks: openjdk:17-jdk-slim
→ Doesn't verify it exists with DockerImageSearch
→ Build fails: IMAGE_PULL (image doesn't exist)
→ Wastes iteration
```

**Missing**: Reminder to ALWAYS verify images before using them

---

## 🔧 Improvements Needed

### Improvement #1: Add PLATFORM_INCOMPATIBLE Handler (URGENT!)

```python
elif stage == "PLATFORM_INCOMPATIBLE":
    self.log(repo_name, "PLATFORM_INCOMPATIBLE - image doesn't support architecture", to_console=True)

    # Extract current image
    current_image = extract_from_dockerfile(dockerfile_path)

    refinement_query = f"""PLATFORM INCOMPATIBILITY ERROR

The Docker image you selected does NOT support {platform_info['display_name']} architecture!

Current image: {current_image}
Your system: {platform_info['display_name']} ({platform_info['description']})
Required: Image must support {platform_info['docker_arch']}

**MANDATORY STEPS**:
**STEP 1**: Use DockerImageSearch with 'tags:{base_image_name}'
   - Look for tags with [OK] marker (architecture compatible)
   - Check "Architectures" field in results

**STEP 2**: Pick a tag that shows {platform_info['docker_arch']} support
   - Prefer: -slim or -bookworm variants (Debian-based, better compatibility)
   - AVOID: -alpine variants (often lack ARM64 support)

**STEP 3**: Verify with DockerImageSearch '{image}:{tag}'
   - Confirm it supports {platform_info['docker_arch']}

**STEP 4**: Update Dockerfile FROM line

**CRITICAL RULES**:
- DO NOT use alpine images if you're on ARM64 (they often don't support it)
- MUST verify with DockerImageSearch BEFORE using
- The [OK] marker means it's compatible with your architecture

Current Dockerfile: {dockerfile_absolute}
Error: {safe_error}

OUTPUT: Updated Dockerfile with ARM64-compatible image
"""
```

### Improvement #2: Strengthen FILE_COPY_MISSING Handler

**Current** (line 1249-1260):
```python
refinement_query = f"""FILE COPY MISSING ERROR

COPY command references file that doesn't exist in build context.

**STEP 1:** Use GrepFiles to find the file in the repository
**STEP 2:** Read current Dockerfile at: {dockerfile_absolute}
**STEP 3:** Fix COPY path or remove if file doesn't exist
**STEP 4:** Check if file is in .dockerignore

DO NOT CHANGE BASE IMAGE! Final Answer: ONLY Dockerfile content.

Error: {safe_error}"""
```

**Improved**:
```python
refinement_query = f"""CRITICAL: FILE COPY ERROR - Path Doesn't Exist!

Your Dockerfile has a COPY command for a path that DOES NOT EXIST in the repository!

**THIS IS NOT AN IMAGE PROBLEM! DO NOT CHANGE THE BASE IMAGE!**

**MANDATORY PROCESS**:
**STEP 1**: Read Dockerfile: {dockerfile_absolute}
   - Identify which COPY command is failing
   - Look for: COPY <source_path> <dest_path>

**STEP 2**: Use DirectoryTree to see repository structure
   - Find what paths ACTUALLY exist
   - Compare with what Dockerfile is trying to COPY

**STEP 3**: Fix the COPY command:
   Option A: Path is wrong → Fix path to correct location
   Option B: File doesn't exist → Remove COPY command
   Option C: Need whole project → Change to COPY . /app
   Option D: File in .dockerignore → Remove from .dockerignore

**EXAMPLE**:
If Dockerfile has: COPY src /app/src
But directory tree shows src/ doesn't exist
→ Either remove COPY line or change to copy what exists

**CRITICAL WARNINGS**:
❌ DO NOT change the base image (this won't fix the problem!)
❌ DO NOT install build tools (this is a path issue, not dependencies!)
❌ DO NOT search for solutions (just fix the path!)

The base image is CORRECT. The COPY path is WRONG.

Error shows: {safe_error}

OUTPUT: Fixed Dockerfile with correct COPY paths (base image unchanged)
"""
```

### Improvement #3: Add Image Verification Reminder to Query 4

**Add to run_agent.py Query 4** (after the examples):
```python
**CRITICAL FINAL CHECK BEFORE USING ANY IMAGE**:

After selecting an image, you MUST verify it:

1. Check architecture compatibility:
   - Look for [OK] marker in DockerImageSearch 'tags:' results
   - Verify with DockerImageSearch '{image}:{tag}'
   - Confirm "Architectures" includes {host_arch}

2. Avoid common mistakes:
   - alpine variants often lack ARM64 support → prefer -slim or -bookworm
   - Old patch versions (3.12.1) may not exist → use MAJOR.MINOR (3.12)
   - Random tags without verification → ALWAYS verify first!

3. If verification fails:
   - Image doesn't exist → Pick different tag from list
   - No ARM64 support → Pick different variant (slim vs alpine)
   - Don't guess! Use the tool!
```

### Improvement #4: Add Multi-Issue Detection

When FILE_COPY_MISSING persists for 3+ iterations:
```python
# After iteration 3 with same FILE_COPY_MISSING
if stage == "FILE_COPY_MISSING" and consecutive_same_error >= 3:
    refinement_query = f"""CRITICAL: YOU'VE FAILED 3 TIMES ON THE SAME ERROR!

You keep getting FILE_COPY_MISSING for the same path!

**STOP CHANGING THE BASE IMAGE - THAT'S NOT THE PROBLEM!**

The issue is: Your COPY command references a path that DOESN'T EXIST!

**What you need to do**:
1. Read the Dockerfile: {dockerfile_absolute}
2. Find the COPY line that's failing
3. Use DirectoryTree to see what paths exist
4. FIX THE COPY COMMAND (not the base image!)

**Examples of fixes**:
- COPY src /app → Change to COPY dubbo-core /app (if that's what exists)
- COPY src /app → Remove it if src truly doesn't exist
- COPY src /app → Change to COPY . /app (copy everything)

The base image is FINE. Fix the COPY path!
"""
```

---

## 📊 Expected Impact

| Issue | Before | After |
|-------|--------|-------|
| PLATFORM_INCOMPATIBLE | Misclassified → wrong prompt | Specific handler → DockerImageSearch for compatible tag |
| FILE_COPY_MISSING persistence | Agent ignores "don't change image" | Stronger warnings + multi-iteration check |
| Image verification | Optional | Mandatory with checklist |
| alpine on ARM64 | Agent keeps trying | Explicit warning to avoid |

---

## 🎯 Priority

1. **URGENT**: Add PLATFORM_INCOMPATIBLE handler (affects many repos)
2. **HIGH**: Strengthen FILE_COPY_MISSING handler (dubbo, keras, etc.)
3. **MEDIUM**: Add image verification reminder to Query 4
4. **LOW**: Add multi-iteration same-error detection

---

## 💡 Key Insight

**The agent needs VERY EXPLICIT instructions** when errors occur:

- ❌ "Fix the issue" → Too vague, agent guesses
- ✅ "COPY path doesn't exist. Use DirectoryTree. Fix the path. DO NOT change image." → Specific, actionable

**The agent is literal**:
- If we say "might need to change image", it will
- If we say "DO NOT change image", it still might if not emphatic enough
- Need to repeat critical points multiple times

**The discovery-based prompts work** for initial generation, but **error recovery needs to be MORE directive**.

---

## 🚀 Next Steps

1. Implement PLATFORM_INCOMPATIBLE handler
2. Strengthen FILE_COPY_MISSING prompt
3. Test on dubbo specifically
4. Add multi-iteration detection for stubborn errors
5. Monitor if success rate improves
