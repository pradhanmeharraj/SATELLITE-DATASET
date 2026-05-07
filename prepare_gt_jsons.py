"""
Create per-image ground-truth JSON files from ground_truth.csv for each sequence.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data" / "synthetic_cubesat"

for seq_dir in DATA_ROOT.iterdir():
    if not seq_dir.is_dir():
        continue
    csv_file = seq_dir / "ground_truth.csv"
    if not csv_file.exists():
        continue
    gt_folder = seq_dir / "ground_truth"
    gt_folder.mkdir(exist_ok=True)
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get('IMG_NUM') or row.get('img') or row.get('image')
            if not img_name:
                continue
            stem = Path(img_name).stem
            # translation
            try:
                tx = float(row.get('X', 0.0))
                ty = float(row.get('Y', 0.0))
                tz = float(row.get('Z', 0.0))
            except ValueError:
                tx = ty = tz = 0.0
            # quaternion columns may be Q1,Q2,Q3,W or x,y,z,w
            qx = row.get('Q1') or row.get('qx') or row.get('x') or row.get('QX')
            qy = row.get('Q2') or row.get('qy') or row.get('y') or row.get('QY')
            qz = row.get('Q3') or row.get('qz') or row.get('z') or row.get('QZ')
            qw = row.get('W') or row.get('qw') or row.get('w')
            try:
                qx = float(qx)
                qy = float(qy)
                qz = float(qz)
                qw = float(qw)
            except Exception:
                # Fallback defaults
                qx = qy = qz = 0.0
                qw = 1.0
            # evaluation expects quaternion as (w,x,y,z)
            quat = [qw, qx, qy, qz]
            out = {
                "translation": [tx, ty, tz],
                "quaternion": quat,
            }
            out_file = gt_folder / f"{stem}.json"
            with open(out_file, 'w', encoding='utf-8') as of:
                json.dump(out, of)
    print(f"Created JSON ground-truth files in {gt_folder}")
