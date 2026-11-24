#!/usr/bin/env python3
"""Send one sample telemetry POST. No packages required."""

import argparse
import json
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("url", nargs="?", default="https://httpbin.org/post")
args = parser.parse_args()

try:
    request = urllib.request.Request(
        args.url,
        data=json.dumps({"probe": True, "gpu_util_pct": 42}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        print(f"HTTP {response.status}")
        print(response.read(4096).decode(errors="replace"))
except urllib.error.HTTPError as error:
    print(f"HTTP {error.code}: server or proxy rejected the request.")
    error.close()
    raise SystemExit(1)
except (urllib.error.URLError, OSError, ValueError) as error:
    print(f"Connection failed: {error}")
    raise SystemExit(1)
