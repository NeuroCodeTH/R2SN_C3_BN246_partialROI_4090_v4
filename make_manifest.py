#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Match C3 original T1 study codes to patient-ID MNI lesion masks.
Pure standard library; will NOT move, copy, or modify input files.
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

DEFAULT_T1 = '/data02/tanghui/C3_T1'
DEFAULT_LESION = '/data02/tanghui/C3_lesion_newversion/C3_img_aftercheck/LesionMNI1mm'
DEFAULT_SCRIPTS = '/data02/tanghui/C3_R2SN/scripts'


def choose_t1(folder):
    if not folder.is_dir():
        return None, 'T1_FOLDER_MISSING'
    cand = sorted(p for p in folder.iterdir() if p.is_file() and (p.name.endswith('.nii.gz') or p.suffix == '.nii') and re.search(r'T1w', p.name, re.I))
    if not cand:
        return None, 'T1_NOT_FOUND'
    if len(cand) != 1:
        return None, 'MULTIPLE_T1'
    return cand[0], 'MATCHED'


def make(t1_root, id_table, lesion_root, manifest, report):
    t1_root = Path(t1_root)
    id_table = Path(id_table)
    lesion_root = Path(lesion_root)
    for p in (t1_root, id_table, lesion_root):
        if not p.exists():
            raise FileNotFoundError(f'输入不存在: {p}')

    # Filename example: 1001102379_Lesion2MNI1mmNL.nii.gz
    lesion_index = {}
    duplicate_lesions = set()
    for f in lesion_root.iterdir():
        if not f.is_file():
            continue
        m = re.fullmatch(r'(\d+)_Lesion2MNI1mmNL\.nii(?:\.gz)?', f.name, re.I)
        if not m:
            continue
        sid = m.group(1)
        if sid in lesion_index:
            duplicate_lesions.add(sid)
        lesion_index[sid] = f

    raw = []
    for n, line in enumerate(id_table.read_text(encoding='utf-8-sig').splitlines(), 1):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = re.split(r'\s+', s)
        if len(parts) < 2:
            raw.append(dict(line=n, code=s, id='', issue='ID_TABLE_BAD_LINE'))
            continue
        code, sid = parts[:2]
        if not re.fullmatch(r'\d{3}-\d{4,}', code) or not sid.isdecimal():
            # Allow header to be skipped explicitly rather than treating as patient
            if n == 1 and (not sid.isdecimal()):
                continue
            raw.append(dict(line=n, code=code, id=sid, issue='ID_TABLE_BAD_CODE_OR_ID'))
            continue
        raw.append(dict(line=n, code=code, id=sid, issue=''))

    id_counts = Counter(r['id'] for r in raw if not r['issue'])
    code_counts = Counter(r['code'] for r in raw if not r['issue'])
    matched = []
    checked = []
    for r in raw:
        code, sid = r['code'], r['id']
        folder = t1_root / code[:3] / code if not r['issue'] else None
        tp, status = (None, r['issue']) if r['issue'] else choose_t1(folder)
        if not r['issue'] and (id_counts[sid] > 1 or code_counts[code] > 1):
            status = 'DUPLICATE_ID_OR_CODE'
        if not r['issue'] and sid in duplicate_lesions:
            status = 'DUPLICATE_LESIONS'
        lesion = lesion_index.get(sid)
        if not r['issue'] and status == 'MATCHED' and lesion is None:
            status = 'LESION_NOT_FOUND'
        row = dict(line=r['line'], code=code, id=sid,
                   t1=str(tp) if tp else '', lesion=str(lesion) if lesion else '',
                   status=status)
        checked.append(row)
        if status == 'MATCHED':
            matched.append(dict(id=sid, code=code, t1=str(tp), lesion=str(lesion), lesion_space='mni'))

    manifest = Path(manifest)
    report = Path(report)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['id', 'code', 't1', 'lesion', 'lesion_space'])
        wr.writeheader()
        wr.writerows(matched)
    with report.open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['line', 'code', 'id', 't1', 'lesion', 'status'])
        wr.writeheader()
        wr.writerows(checked)

    issues = Counter(r['status'] for r in checked)
    extra = len(set(lesion_index) - {r['id'] for r in raw})
    print(f'ID表行数: {len(raw)}; 病灶文件ID数: {len(lesion_index)}')
    print(f'可配对T1+MNI病灶: {len(matched)}; 表外病灶ID数: {extra}')
    for key, val in sorted(issues.items()):
        print(f'{key}: {val}')
    print(f'MANIFEST: {manifest}\nREPORT: {report}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t1-root', default=DEFAULT_T1)
    p.add_argument('--id-table', default=f'{DEFAULT_T1}/code_id_all.txt')
    p.add_argument('--lesion-root', default=DEFAULT_LESION)
    p.add_argument('--manifest', default=f'{DEFAULT_SCRIPTS}/C3_MNI_manifest.csv')
    p.add_argument('--report', default=f'{DEFAULT_SCRIPTS}/C3_match_report.csv')
    a = p.parse_args()
    make(a.t1_root, a.id_table, a.lesion_root, a.manifest, a.report)

if __name__ == '__main__':
    main()
