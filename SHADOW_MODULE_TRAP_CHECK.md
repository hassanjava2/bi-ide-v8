# SHADOW MODULE TRAP CHECK

## Objective
Ensure synchronized updates for duplicate `hierarchy/__init__.py` locations as required by rules.

## Paths To Verify
1. `/home/bi/bi-ide-v8/hierarchy/__init__.py`
2. `/home/bi/.bi-ide-worker/hierarchy/__init__.py`

## Check Items
- [ ] Both files updated when hierarchy entrypoints change
- [ ] Version/hash parity recorded
- [ ] No drift detected after deploy

## Verification Record
- Date:
- Commit/Change ref:
- Hash path 1:
- Hash path 2:
- Result: PASS/FAIL

## Notes
- Any mismatch = release block until resolved.
