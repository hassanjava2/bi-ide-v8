# SERVICES CONTINUITY CHECKLIST

**Window:** 24h continuous monitoring  
**Date:** 2026-03-05

## Required Services
- [ ] RTX API Server (8090)
- [ ] Training Daemon (`auto_training_daemon.py`)
- [ ] Bulk Downloader (`bulk_downloader.py`)
- [ ] Knowledge Scout (`knowledge_scout.py`)
- [ ] Ollama (training-only)

## Monitoring Checks
- [ ] Service process alive every interval
- [ ] Port/listener healthy where applicable
- [ ] No repeated crash-loop
- [ ] Logs show normal activity

## Evidence
- Monitor script/log path:
- Uptime report:
- Alerts summary:

## Result
- [ ] PASS
- [ ] FAIL (with incident link)
