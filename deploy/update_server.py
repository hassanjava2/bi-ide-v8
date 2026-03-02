#!/usr/bin/env python3
"""
Standalone Update Manifest Server for BI-IDE Desktop Auto-Update.
Runs on port 8011 — nginx proxies /api/v1/updates/* here.
"""
import json, os, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

LATEST_VERSION = "8.0.1"
RELEASE_NOTES = "v8.0.1 — AI Chat fix, Real Scouts, Auto-Update, RTX Training"
DOWNLOAD_URL = "https://github.com/hassanjava2/bi-ide-v8/releases/latest"
LOG_DIR = "/opt/bi-iq-app/logs/updates"
os.makedirs(LOG_DIR, exist_ok=True)

def compare_versions(current, latest):
    try:
        c = [int(x) for x in current.strip().split(".")]
        l = [int(x) for x in latest.strip().split(".")]
        return l > c
    except: return False

class UpdateHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/api/v1/updates/manifest", "/manifest", "/"):
            params = parse_qs(parsed.query)
            cv = params.get("current_version", ["0.0.0"])[0]
            ch = params.get("channel", ["stable"])[0]
            has = compare_versions(cv, LATEST_VERSION)
            r = {"has_update": has, "current_version": cv, "latest_version": LATEST_VERSION, "channel": ch}
            if has:
                r.update({"version": LATEST_VERSION, "critical": False, "size_mb": 45.0,
                           "estimated_download_size_mb": 45.0, "download_url": DOWNLOAD_URL,
                           "release_notes": RELEASE_NOTES})
            self._json(200, r)
        elif parsed.path == "/health":
            self._json(200, {"status": "ok", "service": "update-server", "version": LATEST_VERSION})
        else:
            self._json(404, {"detail": "Not Found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/api/v1/updates/report", "/report"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode() if length else "{}"
            with open(f"{LOG_DIR}/update_reports.jsonl", "a") as f:
                f.write(json.dumps({"data": body, "ts": time.time()}) + "\n")
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"detail": "Not Found"})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args): pass  # Silence logs

if __name__ == "__main__":
    print(f"Update server v{LATEST_VERSION} on :8011")
    HTTPServer(("127.0.0.1", 8011), UpdateHandler).serve_forever()
