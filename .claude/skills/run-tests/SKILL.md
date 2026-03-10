---
name: run-tests
description: Runs the project test suite and reports results. Use when the user asks to run tests, check if things work, or verify changes. Don't use for writing new tests — use add-endpoint or review-security for that.
allowed-tools: Bash(cd * && * pytest *), Read, Grep
---

Run the backend test suite and report results clearly.

## Steps

1. Activate the backend venv and run pytest:
   ```bash
   cd backend && source .venv312/bin/activate && python -m pytest test_security.py -v 2>&1
   ```

2. Parse the output:
   - Count passed/failed/errors
   - If any failures: read the failing test, identify the root cause, and suggest a fix
   - If all pass: confirm with a summary

3. If the user made recent changes, check if new tests are needed:
   - Read the git diff for changed files
   - If `security.py` or `server.py` changed, verify test coverage for the changes
   - Suggest specific new test cases if coverage gaps exist

## Output Format

Report as:
- Total: X tests
- Passed: X | Failed: X | Errors: X
- If failures: show the test name, what failed, and a suggested fix
