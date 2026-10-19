# Setup: Kaggle Notebook settings -> Accelerator: GPU T4 x2, Internet: ON.
# Everything is written to /kaggle/working (download the zip of outputs at the end).

# %% [code]
# Install + download FOD-A (Pascal VOC v2.1, official repo: github.com/FOD-UNOmaha/FOD-data)
import subprocess, sys, zipfile
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "gdown"], check=True)
import gdown
Path("/kaggle/working/raw").mkdir(parents=True, exist_ok=True)
zip_path = gdown.download(id="1RdErcq8PGRXZUOGauaACkQG44T-QyZ4x", output="/kaggle/working/raw/fod_voc.zip", quiet=False)
assert zip_path and Path(zip_path).exists(), "download failed - check Internet is ON (and retry; Google Drive may rate-limit)"
zipfile.ZipFile(zip_path).extractall("/kaggle/working/raw")
print("extracted to /kaggle/working/raw:", [q.name for q in Path("/kaggle/working/raw").iterdir()])

# %% [code]
# Prepare dataset: VOC -> YOLO, clean, stratified block-wise subset + split
import hashlib, random, re, shutil, os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/kaggle/working")
ROOT = next(Path("raw").rglob("VOC2007"))
DST = Path("data")
FRAC = 0.12         # fraction of each class's frame-blocks (1.0 = full data).
BLOCK, SEED = 20, 42   # small blocks -> finer, still leakage-free split on a small subset

all_xmls = sorted((ROOT / "Annotations").glob("*.xml"))
obj_re = re.compile(r"<object>\s*<name>([^<]+)</name>")
file_cls = {x: obj_re.findall(x.read_text()) for x in all_xmls}
names = sorted({n for v in file_cls.values() for n in v})
cid = {n: i for i, n in enumerate(names)}

# Frames are stored grouped by class/video: split whole blocks per class so neighbouring
# (near-identical) frames never straddle train/val/test and every class is in every split.
blk_cls = {}
for x, v in file_cls.items():
    blk_cls.setdefault(int(x.stem) // BLOCK, Counter()).update(v)
by_cls = {}
for blk, c in blk_cls.items():
    by_cls.setdefault(c.most_common(1)[0][0], []).append(blk)
split_of = {}
for c, blks in by_cls.items():
    blks.sort(); random.Random(SEED).shuffle(blks)
    blks = blks[:max(4, round(FRAC * len(blks)))]
    n = len(blks); n_val = max(1, round(.15 * n)); n_test = max(1, round(.15 * n))
    for i, blk in enumerate(blks):
        split_of[blk] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
xmls = [x for x in all_xmls if int(x.stem) // BLOCK in split_of]

issues, seen, keep = Counter(), set(), []
for x in xmls:
    im = ROOT / "JPEGImages" / f"{x.stem}.jpg"
    img = cv2.imread(str(im)) if im.exists() else None
    if img is None:
        issues["bad_image"] += 1; continue
    md5 = hashlib.md5(im.read_bytes()).hexdigest()
    if md5 in seen:
        issues["duplicate"] += 1; continue
    seen.add(md5)
    h, w = img.shape[:2]; rows = []
    for o in ET.parse(x).getroot().iter("object"):
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            issues["degenerate_box"] += 1; continue
        rows.append((cid[o.findtext("name")], (x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h))
    if rows:
        keep.append((int(x.stem), im, rows))
print("used", len(keep), "of", len(all_xmls), "| issues", dict(issues), "| classes", len(names))

cls_count = Counter(r[0] for _, _, rows in keep for r in rows)
areas = [r[3] * r[4] for _, _, rows in keep for r in rows]
print("median box area frac %.4f | <1%% of image: %.1f%%" % (
    sorted(areas)[len(areas)//2], 100 * sum(a < .01 for a in areas) / len(areas)))
Path("report_assets").mkdir(exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
order = sorted(cls_count, key=lambda c: -cls_count[c])
ax[0].bar([names[c] for c in order], [cls_count[c] for c in order]); ax[0].tick_params(axis="x", rotation=90)
ax[0].set_title("Instances per class"); ax[1].hist(areas, bins=60); ax[1].set_xscale("log")
ax[1].set_title("Box area / image area"); plt.tight_layout(); plt.savefig("report_assets/eda.png", dpi=120)

if DST.exists(): shutil.rmtree(DST)
for s in ("train", "val", "test"):
    (DST/"images"/s).mkdir(parents=True); (DST/"labels"/s).mkdir(parents=True)
n_img, n_inst = Counter(), Counter()
for n, im, rows in keep:
    s = split_of[n // BLOCK]
    shutil.copy(im, DST/"images"/s/im.name)
    (DST/"labels"/s/f"{im.stem}.txt").write_text("\n".join("%d %.6f %.6f %.6f %.6f" % r for r in rows))
    n_img[s] += 1; n_inst[s] += len(rows)
print("images per split:", dict(n_img), "| instances:", dict(n_inst))
Path("data.yaml").write_text(
    "path: /kaggle/working/data\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    + "".join(f"  {i}: {n}\n" for n, i in cid.items()))

# %% [code]
# Train (YOLO11s @512 on a Kaggle GPU
from ultralytics import YOLO
model = YOLO("yolo11s.pt")           # COCO-pretrained: transfer learning for a small dataset
model.train(
    data="/kaggle/working/data.yaml",
    imgsz=512,                       # images are 300x300; 512 (~1.7x upsample) keeps small objects visible, cheaper than 640
    epochs=15, patience=6, batch=32, workers=4, cache="ram",   # cache="ram": decode JPEGs once, removes the CPU data-loading bottleneck
    optimizer="AdamW", lr0=0.001, cos_lr=True, pretrained=True,
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,       # lighting / weather / time of day
    degrees=15, flipud=0.5, fliplr=0.5,       # top-down UAV view: no canonical "up"
    scale=0.6,                                # altitude / zoom variation
    mosaic=1.0, close_mosaic=3, mixup=0.1, translate=0.1, perspective=0.0005,
    project="/kaggle/working/runs", name="fod_yolo11s", seed=42,
)
WEIGHTS = "/kaggle/working/runs/fod_yolo11s/weights/best.pt"

# %% [code]
# Test-set metrics (mAP50, mAP50-95, P, R, PR curve, confusion matrix)
model = YOLO(WEIGHTS)
m = model.val(data="/kaggle/working/data.yaml", split="test", imgsz=512, conf=0.001, plots=True,
              project="/kaggle/working/runs", name="test_eval")
print(f"mAP@50={m.box.map50:.3f}  mAP@50-95={m.box.map:.3f}  P={m.box.mp:.3f}  R={m.box.mr:.3f}")
# Plots are in /kaggle/working/runs/test_eval: PR_curve.png, F1_curve.png, confusion_matrix*.png
# Pick the operating threshold from F1_curve.png / P_curve.png / R_curve.png (recall-biased for safety).

# %% [code]
# TP / FP / FN gallery at the chosen threshold
CONF = 0.25   # <- change after looking at the curves

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2-x1) * max(0, y2-y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u else 0

out = Path("report_assets/gallery"); out.mkdir(parents=True, exist_ok=True)
stats = []
for p in sorted(Path("data/images/test").glob("*.jpg")):
    img = cv2.imread(str(p)); h, w = img.shape[:2]
    gt = []
    for line in (Path("data/labels/test") / f"{p.stem}.txt").read_text().splitlines():
        c, x, y, bw, bh = line.split(); c = int(c); x, y, bw, bh = map(float, (x, y, bw, bh))
        gt.append((c, [(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h]))
    r = model.predict(img, conf=CONF, imgsz=512, verbose=False)[0]
    preds = sorted(zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()), key=lambda t: -t[1])
    matched, tp, fp = set(), 0, 0
    for c, cf, b in preds:
        hit = next((i for i, (gc, gb) in enumerate(gt) if i not in matched and gc == c and iou(b, gb) >= .5), None)
        col = (0, 200, 0) if hit is not None else (0, 0, 255)
        if hit is not None: matched.add(hit); tp += 1
        else: fp += 1
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{'TP' if hit is not None else 'FP'} {model.names[c]} {cf:.2f}", (x1, max(12, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, col, 1)
    fn = [g for i, g in enumerate(gt) if i not in matched]
    for c, b in fn:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"FN {model.names[c]}", (x1, min(h-4, y2+14)), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 0, 0), 1)
    stats.append((p.name, tp, fp, len(fn)))
    cv2.imwrite(str(out / f"{'fn' if fn else 'fp' if fp else 'tp'}_{p.name}"), img)
TP, FP, FN = (sum(s[i] for s in stats) for i in (1, 2, 3))
print(f"@conf={CONF}: TP={TP} FP={FP} FN={FN}  precision={TP/max(TP+FP,1):.3f}  recall={TP/max(TP+FN,1):.3f}")
print({k: len(list(out.glob(k + '_*'))) for k in ('tp', 'fp', 'fn')})

# %% [code]
# Result video (test frames in order, boxes + labels + confidence), plus per-image montage
files = sorted(Path("data/images/test").glob("*.jpg"))[::3][:600]          # every 3rd frame, up to 600
size = 600
vw = cv2.VideoWriter("/kaggle/working/result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
for p in files:
    f = model.predict(cv2.imread(str(p)), conf=CONF, imgsz=512, verbose=False)[0].plot()
    vw.write(cv2.resize(f, (size, size)))
vw.release()
print("video saved: /kaggle/working/result.mp4")

# %% [code]
# Bundle outputs for download
import shutil
from pathlib import Path
os.chdir("/kaggle/working")
bundle = Path("bundle"); shutil.rmtree(bundle, ignore_errors=True); bundle.mkdir()
for src in ("report_assets", "runs/test_eval", "data.yaml", "result.mp4",
            "runs/fod_yolo11s/weights/best.pt", "runs/fod_yolo11s/results.csv", "runs/fod_yolo11s/results.png"):
    dst = bundle / src
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).is_dir(): shutil.copytree(src, dst)
    elif Path(src).exists(): shutil.copy(src, dst)
shutil.make_archive("/kaggle/working/fod_outputs", "zip", bundle)
print("created /kaggle/working/fod_outputs.zip")
