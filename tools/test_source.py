"""Emits a synthetic NDI source (black background + white 'caption' rectangle) to
test the main pipeline (src/main.py) without needing a real subtitle generator.

Usage:
    python3 tools/test_source.py
Then, in another terminal:
    python3 src/main.py --source TEST-CAPTION-SRC --output-name "AlphaSubs"
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ndi_io  # noqa: E402


def main():
    ndi_io.initialize()
    sender = ndi_io.Sender("TEST-CAPTION-SRC")
    h, w = 480, 854
    frame = np.zeros((h, w, 4), dtype=np.uint8)
    frame[:, :, 3] = 255
    frame[200:260, 300:560, :3] = 255  # white block simulating text

    print("Sending NDI 'TEST-CAPTION-SRC'. Ctrl+C to stop.")
    try:
        while True:
            sender.send(frame)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
        ndi_io.shutdown()


if __name__ == "__main__":
    main()
