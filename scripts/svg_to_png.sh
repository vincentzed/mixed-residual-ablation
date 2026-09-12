#!/usr/bin/env bash
# Rasterize <dir>/*.svg to <dir>/*.png via headless Chromium (playwright's —
# works on Linux boxes with no system Chrome, and on macOS), then autocrop
# on the alpha channel with Pillow. Retina-crisp via device_scale_factor=2.
#
# The page is screenshotted with omit_background=True, so the result is a
# rounded card with a TRANSPARENT outside — matching the vega-lite cards.
#
# Usage: svg_to_png.sh [dir]   (default: ./media next to the calling repo)
set -euo pipefail
cd "${1:-$(dirname "$0")/../media}"

uv run --no-project --with playwright --with pillow python - <<'EOF'
import glob
import os
import subprocess
import sys

from playwright.sync_api import sync_playwright

subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"],
               check=True, capture_output=True)

from PIL import Image  # noqa: E402

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 2400},
                            device_scale_factor=2)
    for svg in sorted(glob.glob("*.svg")):
        png = svg[:-4] + ".png"
        page.goto(f"file://{os.getcwd()}/{svg}")
        page.wait_for_timeout(1500)  # let any webfonts settle
        page.screenshot(path=png, omit_background=True)
        img = Image.open(png).convert("RGBA")
        bbox = img.getchannel("A").getbbox()
        if bbox:
            pad = 8
            bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                    min(img.width, bbox[2] + pad),
                    min(img.height, bbox[3] + pad))
            img.crop(bbox).save(png)
        print(png, Image.open(png).size)
    browser.close()
EOF
