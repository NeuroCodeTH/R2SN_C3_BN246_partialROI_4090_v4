#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3 BN246 stroke-adapted R2SN: extract radiomics from REMAINING ROI voxels.

* Only anatomical lesion + optional 1mm margin are removed from each atlas ROI.
* No whole-subject QC rejection due to ROI lesion fraction.
* Completely lesion-erased ROI -> 0 row/column in the 246x246 network.
* Nonempty ROI that fails PyRadiomics -> NaN row/column, NEVER mislabeled as 0.
* Original R2SN PyRadiomics Params.yaml/picked_features.txt, per-feature
  min-max across surviving valid ROIs, pairwise Pearson (official formula).
* T1 originals and lesion originals are never modified/copied.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt
from network_utils import classify_rois, valid_feature_network, nodal_metrics, write_rows, atomic_save_npy

VERSION = 'BN246_remaining_roi_zero_v4.1'
DEFAULT_ROOT = '/data02/tanghui/C3_R2SN'
DEFAULT_TAG = 'lesionaware_BN246_remainingROI_v4'


def package_defaults():
    import R2SN
    root=Path(R2SN.__file__).resolve().parent
    return dict(atlas=root/'data'/'rBN_Atlas_246_1mm.nii',
                target=root/'data'/'MNI152_T1_1mm.nii',
                params=root/'network_feature'/'Params.yaml',
                picked=root/'network_feature'/'picked_features.txt')


def same_grid(a,b,tol=0.02):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine,b.affine,atol=tol,rtol=0)


def dilate_mm(a,affine,mm):
    if mm < 0:raise ValueError('--margin-mm cannot be <0')
    a=np.asarray(a,dtype=bool)
    if mm==0 or not a.any():return a.copy()
    return distance_transform_edt(~a,sampling=nib.affines.voxel_sizes(affine)[:3]) <= mm


def save_nii_like(data,ref,path,dtype=np.uint8):
    header=ref.header.copy();header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(np.asarray(data,dtype=dtype),ref.affine,header),str(path))


def lesion_in_target(path,target):
    orig=nib.load(str(path))
    if len(orig.shape)!=3:raise ValueError('lesion NIfTI is not 3D')
    match=same_grid(orig,target)
    img=orig if match else resample_from_to(orig,target,order=0,mode='constant',cval=0)
    a=np.asarray(img.dataobj)
    if not np.isfinite(a).all():raise ValueError('lesion NIfTI includes NaN/Inf')
    mask=a>0
    if not mask.any():raise ValueError('MNI lesion empty after resampling; check inputs and world coordinates')
    return mask,dict(original_shape=list(orig.shape),resampled=not match,
                     original_affine=np.round(orig.affine,5).tolist())


def register_t1(t1,target,lesion,margin,mask_mode,tmp):
    import ants
    fixed=ants.image_read(str(target))
    moving=ants.image_read(str(t1))
    denoised=ants.denoise_image(moving,ants.get_mask(moving))
    corrected=ants.n4_bias_field_correction(denoised)
    kw=dict(type_of_transform='SyN',outprefix=str(tmp/'ants_'))
    if mask_mode=='fixed':
        ref=nib.load(str(target))
        expanded=dilate_mm(lesion,ref.affine,margin)
        p=tmp/'fixed_lesion_exclusion.nii.gz'
        save_nii_like(expanded,ref,p)
        ex=ants.image_read(str(p))
        brain=ants.get_mask(fixed)
        valid=(np.asarray(brain.numpy())>0) & ~(np.asarray(ex.numpy())>0)
        if valid.sum()<10000:raise ValueError('ANTs fixed registration metric mask too small')
        kw['mask']=ants.from_numpy(valid.astype('float32'),origin=fixed.origin,
                                  spacing=fixed.spacing,direction=fixed.direction)
        kw['mask_all_stages']=True
    result=ants.registration(fixed,corrected,**kw)
    d=np.asarray(result['warpedmovout'].numpy(),dtype=np.float32)
    flat=d.ravel()
    p95=float(np.percentile(flat,95))
    if not np.isfinite(p95) or p95<=0:raise ValueError('invalid registered T1 95th percentile')
    d=d/p95*255.0
    out=tmp/'normalize_255_T1w.nii.gz'
    ants.image_write(ants.from_numpy(d,origin=fixed.origin,spacing=fixed.spacing,direction=fixed.direction),str(out))
    return out


def maybe_reuse_registration(args,target_nii,margin,tmp):
    """Optional reuse only when old run has identical inputs/masking and MNI grid."""
    if not args.reuse_registered_from:return None
    old=Path(args.out_root)/args.id/args.reuse_registered_from
    s=old/'qc'/'run_summary.json'
    t=old/'qc'/'registered_T1_MNI.nii.gz'
    if not (s.is_file() and t.is_file()):return None
    try:
        m=json.loads(s.read_text(encoding='utf-8'))
        checks=dict(t1=args.t1,lesion=args.lesion,
                    registration_mask=args.registration_mask,margin_mm=margin)
        if m.get('status')!='SUCCESS' or any(str(m.get(k))!=str(v) for k,v in checks.items()):
            return None
        if not same_grid(nib.load(str(t)),target_nii):return None
    except Exception:
        return None
    # Safe same-world-grid reuse, only if user explicitly opted in
    return t


def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while True:
            b=f.read(1024*1024)
            if not b:break
            h.update(b)
    return h.hexdigest()


def settings_signature(args,cfg):
    return dict(version=VERSION,id=args.id,code=args.code,t1=str(args.t1),
                lesion=str(args.lesion),nroi=246,tag=args.tag,margin_mm=float(args.margin_mm),
                min_voxels=int(args.min_voxels),registration_mask=args.registration_mask,
                atlas=str(cfg['atlas']),target=str(cfg['target']),
                params_sha256=sha256_file(cfg['params']),picked_sha256=sha256_file(cfg['picked']),
                fully_destroyed_policy='row_col_zero',
                unextractable_nonempty_policy='NaN',
                feature_normalization='minmax_across_surviving_valid_ROIs')


def valid_existing(summary_path,meta,base,sid):
    if not summary_path.is_file():return False
    try:
        prev=json.loads(summary_path.read_text(encoding='utf-8'))
        if prev.get('status') not in ('SUCCESS','SUCCESS_WITH_MISSING_NODES'):return False
        if any(str(prev.get(k))!=str(v) for k,v in meta.items()):
            raise RuntimeError(f'{base}: same --tag has an existing success with DIFFERENT parameters; choose another tag')
        net=base/'network'/f'{sid}_network.npy'
        mask=base/'network'/f'{sid}_valid_roi_mask.npy'
        feat=base/'radiomics'/'ROI_features_246.csv'
        nod=base/'radiomics'/'ROI_node_metrics.csv'
        if not all(f.is_file() and f.stat().st_size>0 for f in [net,mask,feat,nod]):return False
        arr=np.load(net,mmap_mode='r')
        return arr.shape==(246,246)
    except RuntimeError:raise
    except Exception:return False


def run(args):
    from R2SN.network_feature.feature_extract import feature_extract
    defaults=package_defaults()
    cfg={k:Path(getattr(args,k) or defaults[k]) for k in defaults}
    for p in [args.t1,args.lesion,*cfg.values()]:
        if not Path(p).is_file():raise FileNotFoundError(str(p))
    if args.min_voxels<1:raise ValueError('--min-voxels must be >=1')
    base=Path(args.out_root)/args.id/args.tag
    qc=base/'qc';rad=base/'radiomics';nd=base/'network'
    for x in (qc,rad,nd):x.mkdir(parents=True,exist_ok=True)
    summary=qc/'run_summary.json'
    meta=settings_signature(args,cfg)
    if not args.force and valid_existing(summary,meta,base,args.id):
        print(f'[{args.id}] SKIP_EXISTING_SUCCESS',flush=True)
        return 'SKIP_EXISTING_SUCCESS'
    meta['status']='STARTED';meta['time_start']=time.strftime('%F %T')
    try:
        target=nib.load(str(cfg['target']));atlas_img=nib.load(str(cfg['atlas']))
        if not same_grid(target,atlas_img):raise ValueError('atlas and MNI target grids mismatch')
        atlas_f=np.asarray(atlas_img.dataobj)
        if not np.isfinite(atlas_f).all():raise ValueError('atlas has NaN/Inf')
        atlas=np.rint(atlas_f).astype(np.int16)
        if any(not np.any(atlas==i) for i in range(1,247)):
            raise ValueError('atlas missing some labels in 1..246: verify BN246 atlas')
        lesion,details=lesion_in_target(args.lesion,target)
        meta['lesion_mni_info']=details
        meta['lesion_volume_mm3']=float(lesion.sum()*abs(np.linalg.det(target.affine[:3,:3])))
        expanded=dilate_mm(lesion,target.affine,args.margin_mm)
        clean,rows=classify_rois(atlas,lesion,expanded,246,args.min_voxels)
        meta['n_lesioned_rois']=sum(r['lesion_voxels']>0 for r in rows)
        meta['n_fully_lesioned_rois']=sum(r['status']=='FULLY_LESIONED_ZERO' for r in rows)
        meta['n_empty_after_margin']=sum(r['status']=='EMPTY_AFTER_MARGIN_ZERO' for r in rows)
        meta['n_low_voxel_rois']=sum(r['status']=='LOW_VOXELS_ATTEMPT' for r in rows)
        write_rows(qc/'ROI_lesion_overlap.csv',rows)
        if args.keep_qc_images:
            save_nii_like(lesion,target,qc/'lesion_resampled_MNI.nii.gz')
            save_nii_like(clean,target,qc/'BN_lesionExcluded.nii.gz',np.int16)
        with tempfile.TemporaryDirectory(prefix=f'R2SN_partial_{args.id}_',dir=args.tmp_root) as tempdir:
            tmp=Path(tempdir)
            t1reg=maybe_reuse_registration(args,target,args.margin_mm,tmp)
            if t1reg is None:
                t1reg=register_t1(args.t1,cfg['target'],lesion,args.margin_mm,args.registration_mask,tmp)
                meta['registration_reused_from']=''
            else:
                meta['registration_reused_from']=args.reuse_registered_from
            if not same_grid(nib.load(str(t1reg)),target):
                raise ValueError('registered T1 not same voxel grid as MNI target')
            if args.keep_qc_images:
                if Path(t1reg).resolve() != (qc/'registered_T1_MNI.nii.gz').resolve():
                    shutil.copyfile(t1reg,qc/'registered_T1_MNI.nii.gz')
            clean_path=tmp/'BN_lesionExcluded.nii.gz'
            save_nii_like(clean,target,clean_path,np.int16)
            input_csv=tmp/'pyradiomics_input.csv'
            roi_candidates=[r['ROI'] for r in rows if r['clean_voxels']>0]
            with input_csv.open('w',newline='',encoding='utf-8') as f:
                wr=csv.writer(f);wr.writerow(['Image','Mask','Label'])
                for i in roi_candidates:wr.writerow([str(t1reg),str(clean_path),i])
            print(f'[{args.id}] Eligible non-empty ROIs: {len(roi_candidates)}/246; extracting original R2SN features',flush=True)
            rtemp=tmp/'radiomics';rtemp.mkdir()
            if roi_candidates:
                feature_name=feature_extract(str(input_csv),str(cfg['params']),str(rtemp))
                feature_csv=rtemp/feature_name
                feat=pd.read_csv(feature_csv) if feature_csv.is_file() else pd.DataFrame()
            else:
                feat=pd.DataFrame()
            picked=ast.literal_eval(cfg['picked'].read_text(encoding='utf-8'))
            W,valid_mask,full_feat,states,stats=valid_feature_network(feat,picked,rows,246)
            for r,s in zip(rows,states):r['status']=s
            write_rows(qc/'ROI_lesion_overlap.csv',rows)
            meta.update(stats)
            meta['feature_rows_extracted']=int(len(feat))
            meta['feature_rows_requested']=int(len(roi_candidates))
            meta['n_rois_lesion_fraction_over_5pct']=sum(r['lesion_fraction']>0.05 for r in rows)
            meta['n_rois_lesion_fraction_over_20pct']=sum(r['lesion_fraction']>0.20 for r in rows)
            allfeatures=pd.DataFrame(full_feat,columns=picked)
            allfeatures.insert(0,'status',states)
            allfeatures.insert(0,'ROI',list(range(1,247)))
            allfeatures.to_csv(rad/'ROI_features_246.csv',index=False)
            if feature_csv.is_file():shutil.copyfile(feature_csv,rad/'pyradiomics_input_radiomics_features.csv')
            else:pd.DataFrame(columns=['Image','Mask','Label']+picked).to_csv(rad/'pyradiomics_input_radiomics_features.csv',index=False)
            write_rows(rad/'ROI_node_metrics.csv',nodal_metrics(W,valid_mask,rows,states))
            atomic_save_npy(nd/f'{args.id}_network.npy',W)
            atomic_save_npy(nd/f'{args.id}_valid_roi_mask.npy',valid_mask)
            # NaN-to-zero matrix strictly for visualization/software that requires finite numbers.
            # Do NOT treat missing nonempty ROI as biologically destroyed in inference.
            atomic_save_npy(nd/f'{args.id}_network_zeroFilled_DISPLAY_ONLY.npy',np.nan_to_num(W,nan=0.0))
            meta['network_path']=str(nd/f'{args.id}_network.npy')
            meta['radiomics_path']=str(rad/'ROI_features_246.csv')
            if stats['n_other_missing']:
                meta['status']='SUCCESS_WITH_MISSING_NODES'
            else:
                meta['status']='SUCCESS'
            return meta['status']
    except Exception as e:
        meta['status']='FAILED'
        meta['error']=f'{type(e).__name__}: {e}'
        return 'FAILED'
    finally:
        meta['time_end']=time.strftime('%F %T')
        tmp=summary.with_name(summary.name+'.tmp')
        tmp.write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        tmp.replace(summary)
        print(f'[{args.id}] {meta["status"]} valid={meta.get("n_valid",0)} zero={meta.get("n_zero",0)} missing={meta.get("n_other_missing",0)} {meta.get("error","")}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id',required=True);p.add_argument('--code',default='')
    p.add_argument('--t1',required=True);p.add_argument('--lesion',required=True)
    p.add_argument('--out-root',default=DEFAULT_ROOT)
    p.add_argument('--nroi',type=int,choices=[246],default=246)
    p.add_argument('--tag',default=DEFAULT_TAG)
    for k in ('atlas','target','params','picked','tmp_root'):
        p.add_argument('--'+k.replace('_','-'))
    p.add_argument('--registration-mask',choices=['fixed','none'],default='fixed')
    p.add_argument('--margin-mm',type=float,default=0.0)
    p.add_argument('--min-voxels',type=int,default=200,
                   help='QC warning threshold only. Small but nonzero ROIs still ATTEMPT feature extraction.')
    p.add_argument('--keep-qc-images',action='store_true')
    p.add_argument('--reuse-registered-from',default='',help='Optional old tag; reuse only if metadata and grid match.')
    p.add_argument('--force',action='store_true')
    args=p.parse_args()
    status=run(args)
    raise SystemExit(0 if status in ('SUCCESS','SKIP_EXISTING_SUCCESS') else 2 if status=='SUCCESS_WITH_MISSING_NODES' else 1)

if __name__=='__main__':main()
