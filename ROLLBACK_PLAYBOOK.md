# ROLLBACK PLAYBOOK

## Purpose
Standard rollback steps when a release fails health or compliance gates.

## Triggers
1. Error rate above threshold
2. Crash spike after rollout
3. Security gate failure
4. Rule gate failure

## Rollback Steps
1. Stop progressive rollout.
2. Switch traffic/update channel to last known good release.
3. Verify core services health.
4. Run smoke tests on restored version.
5. Log incident + root cause task.

## Verification
- [ ] Application starts
- [ ] Core commands work
- [ ] Sync channel healthy
- [ ] No critical alerts

## Incident Record
- Release:
- Start time:
- Detection:
- Action taken:
- Recovery time:
- Follow-up owner:
