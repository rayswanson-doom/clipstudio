"""CI probe: which OCR backends install and work next to our stack? Prints a report; never fails the job."""
import importlib.metadata as md
import os
import sys
import time

import cv2
import numpy as np

img = np.full((160, 640, 3), 255, np.uint8)
cv2.putText(img, "SALE ENDS TODAY 50% OFF", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)


def ver(n):
    try:
        return md.version(n)
    except Exception:
        return "-"


def size_mb(pkg):
    try:
        d = os.path.dirname(__import__(pkg).__file__)
    except Exception:
        return 0
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) / 1e6


print("python", sys.version.split()[0], "numpy", ver("numpy"), "opencv", ver("opencv-python"), "mediapipe", ver("mediapipe"))
for name in ("paddlepaddle", "paddleocr", "paddlex", "rapidocr", "rapidocr-onnxruntime", "onnxruntime"):
    print(f"  {name}: {ver(name)}")

print("\n== PaddleOCR ==")
try:
    t = time.time()
    import paddleocr
    from paddleocr import PaddleOCR
    print("import ok", paddleocr.__version__, f"{time.time()-t:.1f}s")
    t = time.time()
    try:  # 3.x
        eng = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        res = eng.predict(img)
        texts = [x for r in res for x in (r.get("rec_texts") if hasattr(r, "get") else r["rec_texts"])]
    except TypeError:  # 2.x
        eng = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)
        res = eng.ocr(img, cls=False)
        texts = [ln[1][0] for page in res if page for ln in page]
    print("RESULT paddleocr:", texts, f"{time.time()-t:.1f}s")
    print("paddle dir MB:", round(size_mb("paddle")), "paddleocr dir MB:", round(size_mb("paddleocr")))
    home = os.path.expanduser("~")
    for root in (os.path.join(home, ".paddlex"), os.path.join(home, ".paddleocr")):
        if os.path.isdir(root):
            tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(root) for f in fs) / 1e6
            print("model cache", root, round(tot, 1), "MB")
except Exception as exc:
    print("PADDLEOCR FAILED:", type(exc).__name__, str(exc)[:400])

print("\n== RapidOCR ==")
for mod in ("rapidocr_onnxruntime", "rapidocr"):
    try:
        t = time.time()
        m = __import__(mod)
        eng = m.RapidOCR()
        out = eng(img)
        if mod == "rapidocr_onnxruntime":
            texts = [x[1] for x in (out[0] or [])]
        else:
            texts = list(getattr(out, "txts", None) or [])
        print(f"RESULT {mod}:", texts, f"{time.time()-t:.1f}s", "pkg MB:", round(size_mb(mod)))
    except Exception as exc:
        print(f"{mod} FAILED:", type(exc).__name__, str(exc)[:300])
