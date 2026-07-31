"""
Quran-ARS test client — for the backend team.

Two modes:

  sync  : POST /grade_recitation with a LOCAL audio file (multipart). Simplest smoke test;
          no webhook needed. Prints score, per-word analysis, and the new tajweed/harakat feedback.

  async : The REAL integration — POST /api/evaluate (JSON, Bearer auth). The service replies
          instantly with a jobId, processes in the background, and POSTs the result to a
          webhookUrl. This script starts a tiny local webhook receiver, fires the request, waits
          for the callback, and prints the full `data` payload (incl. userRecitationDiacritized
          and harakatErrors).

Examples:
  # sync (server running locally, a local wav/ogg/webm file):
  python test_client.py sync --file test_audio.ogg --surah 112

  # async (server running locally; give a PUBLIC audio URL the server can download):
  python test_client.py async --audio-url https://your-cdn/clip.webm --surah 112 \
         --from-ayah 1 --to-ayah 4 --api-key "$AI_API_KEY"

Requires: requests  (pip install requests).  Everything else is stdlib.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass


# ----------------------------------------------------------------------------- pretty printers
def _print_words(words):
    print("\n  Per-word analysis:")
    print("  " + "-" * 70)
    print(f"  {'#':<3} {'word':<14} {'expected':<14} {'ok':<3} {'error':<12} {'time'}")
    print("  " + "-" * 70)
    for i, w in enumerate(words, 1):
        ok = "Y" if w.get("is_correct", w.get("isCorrect")) else "N"
        ts = w.get("timestamp_start", w.get("timestampStart"))
        tstr = f"{ts:.2f}s" if ts is not None else "-"
        err = w.get("error_type_ar") or w.get("errorTypeAr") or "-"
        print(f"  {i:<3} {str(w.get('word') or '-'):<14} {str(w.get('expected') or '-'):<14} "
              f"{ok:<3} {err:<12} {tstr}")
    print("  " + "-" * 70)


def _print_harakat(diacritized, checked, errors):
    print("\n  Recitation with tashkeel (what the learner ACTUALLY said):")
    print(f"    {diacritized}")
    print(f"\n  Tajweed / harakat — checked {checked} word(s), {len(errors)} flagged:")
    if not errors:
        print("    ✓ no harakat errors")
    for e in errors:
        word = e.get("word"); exp = e.get("expected_word") or e.get("expectedWord")
        for d in e.get("details", []):
            print(f"    ✗ {word}  (expected {exp}): letter '{d['letter']}' "
                  f"said {d['got']}, expected {d['expected']}")


# ----------------------------------------------------------------------------- sync mode
def run_sync(args):
    url = args.url or f"{args.base}/grade_recitation"
    if not os.path.exists(args.file):
        sys.exit(f"audio file not found: {args.file}")
    print(f"POST {url}  (file: {args.file})")
    with open(args.file, "rb") as f:
        data = {"target_ayah": args.text}
        if args.surah: data["surah_num"] = args.surah
        if args.ayah:  data["ayah_num"] = args.ayah
        try:
            r = requests.post(url, files={"file": f}, data=data, timeout=300)
        except requests.ConnectionError:
            sys.exit(f"cannot connect to {url} — is the server running? (docker compose up / python main.py)")
    if r.status_code != 200:
        sys.exit(f"ERROR {r.status_code}: {r.text}")
    res = r.json()
    print(f"\n  Score: {int(res['accuracy']*100)}%  ({res['raw_score']})  "
          f"— {'PASSED' if res['passed'] else 'NEEDS PRACTICE'}")
    _print_harakat(res.get("user_recitation_diacritized", res.get("user_recitation", "")),
                   res.get("harakat_checked", 0), res.get("harakat_errors", []))
    _print_words(res.get("words", []))
    json.dump(res, open("last_response.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n  full JSON -> last_response.json")


# ----------------------------------------------------------------------------- async mode
_received = {}

class _WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode("utf-8")
        try: _received["payload"] = json.loads(body)
        except Exception: _received["payload"] = {"_raw": body}
        _received["auth"] = self.headers.get("Authorization")
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"ok":true}')
    def log_message(self, *a): pass  # quiet


def run_async(args):
    if not args.audio_url:
        sys.exit("async mode needs --audio-url (a PUBLIC url the server can download)")
    # 1) start the local webhook receiver
    server = HTTPServer(("0.0.0.0", args.webhook_port), _WebhookHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    webhook_url = f"http://{args.webhook_host}:{args.webhook_port}/webhook"
    print(f"webhook receiver listening on {webhook_url}")

    # 2) fire POST /api/evaluate
    url = f"{args.base}/api/evaluate"
    body = {
        "audioUrl": args.audio_url,
        "surahNumber": args.surah or 1,
        "surahName": args.surah_name or "",
        "fromAyah": args.from_ayah or 1,
        "toAyah": args.to_ayah or (args.from_ayah or 1),
        "userId": 42, "recitationId": 1337,
        "webhookUrl": webhook_url,
        "webhookSecret": args.webhook_secret,
    }
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    print(f"POST {url}")
    try:
        r = requests.post(url, json=body, headers=headers, timeout=30)
    except requests.ConnectionError:
        sys.exit(f"cannot connect to {url} — is the server running?")
    print(f"  immediate response {r.status_code}: {r.text}")
    if r.status_code != 200:
        sys.exit("evaluate call rejected (check AI_API_KEY / body).")
    job_id = r.json().get("jobId")

    # 3) wait for the webhook callback
    print(f"  waiting up to {args.timeout}s for the webhook callback (jobId {job_id}) ...")
    t0 = time.time()
    while "payload" not in _received and time.time() - t0 < args.timeout:
        time.sleep(0.5)
    if "payload" not in _received:
        sys.exit("timed out waiting for the webhook — check the server logs and that it can reach "
                 f"{webhook_url} (network/docker).")
    p = _received["payload"]
    print(f"\n  ✓ webhook received (auth header: {_received.get('auth')})")
    if p.get("status") != "success":
        print(f"  status={p.get('status')}  message={p.get('message')}")
        print(json.dumps(p, ensure_ascii=False, indent=2)); return
    d = p["data"]
    print(f"\n  Score: {d['overallScore']}%  — {'PASSED' if d['passed'] else 'NEEDS PRACTICE'}")
    _print_harakat(d.get("userRecitationDiacritized", ""), d.get("harakatChecked", 0),
                   d.get("harakatErrors", []))
    _print_words(d.get("words", []))
    json.dump(p, open("last_webhook.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n  full webhook payload -> last_webhook.json")


# ----------------------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Quran-ARS test client")
    ap.add_argument("mode", choices=["sync", "async"], help="which endpoint to test")
    ap.add_argument("--base", default="http://localhost:8000", help="server base URL")
    ap.add_argument("--url", default=None, help="(sync) override full endpoint URL")
    ap.add_argument("--file", default="test_audio.ogg", help="(sync) local audio file")
    ap.add_argument("--text", default="بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ", help="(sync) expected ayah text")
    ap.add_argument("--audio-url", default=None, help="(async) PUBLIC url of the recording")
    ap.add_argument("--api-key", default=os.environ.get("AI_API_KEY", ""), help="(async) Bearer AI_API_KEY")
    ap.add_argument("--surah", type=int, default=None)
    ap.add_argument("--surah-name", default=None)
    ap.add_argument("--ayah", type=int, default=None, help="(sync) single ayah number")
    ap.add_argument("--from-ayah", type=int, default=None)
    ap.add_argument("--to-ayah", type=int, default=None)
    ap.add_argument("--webhook-host", default="localhost",
                    help="host the SERVER uses to reach this script (e.g. host.docker.internal in Docker)")
    ap.add_argument("--webhook-port", type=int, default=9099)
    ap.add_argument("--webhook-secret", default="test-secret")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()
    (run_sync if args.mode == "sync" else run_async)(args)
