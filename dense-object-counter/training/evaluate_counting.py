"""Evaluate final counts using the SAME ROI, tile and merge pipeline as the app."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from config import settings
from counting.counter import unique_id
from counting.roi import inside_roi_policy,roi_bounds
from detection.factory import create_detector
from detection.tiled_inference import run_inference
from training.dataset_utils import label_path,read_image,read_labels,split_images

def density_bucket(count):
    if count==0:return "0 (negative)"
    for maximum,name in ((20,"1-20"),(50,"21-50"),(100,"51-100"),(200,"101-200")):
        if count<=maximum:return name
    return "201+"

def metrics(rows):
    if not rows:return {"images":0,"mean_absolute_count_error":None,"exact_count_accuracy":None,"mean_signed_error":None}
    n=len(rows); return {"images":n,"mean_absolute_count_error":sum(r["absolute_error"] for r in rows)/n,
                         "exact_count_accuracy":sum(r["error"]==0 for r in rows)/n,"mean_signed_error":sum(r["error"] for r in rows)/n}

def ground_truth_count(boxes,shape,bounds,cfg):
    h,w=shape[:2]; rx1,ry1,rx2,ry2=bounds; count=0
    for a,b,c,d in boxes:
        box=(a*w-rx1,b*h-ry1,c*w-rx1,d*h-ry1)
        keep=inside_roi_policy(box,rx2-rx1,ry2-ry1,cfg) if cfg.EXCLUDE_ROI_EDGE_OBJECTS else (box[2]>0 and box[3]>0 and box[0]<rx2-rx1 and box[1]<ry2-ry1)
        count+=bool(keep)
    return count

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=settings.DATASET_YAML); parser.add_argument("--split",choices=["val","test"],default="test")
    parser.add_argument("--model",type=Path,default=settings.MODEL_PATH); parser.add_argument("--detector",choices=["yolo","opencv"],default=settings.DETECTOR)
    parser.add_argument("--device",default=settings.DEVICE); parser.add_argument("--scope",choices=["roi","full"],default="roi")
    parser.add_argument("--roi-source",choices=["video_stream","still_endpoint"],default="video_stream"); parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args()
    cfg=replace(settings,MODEL_PATH=args.model.resolve(),DETECTOR=args.detector,DEVICE=args.device,ROI_ENABLED=args.scope=="roi",
                EXCLUDE_ROI_EDGE_OBJECTS=settings.EXCLUDE_ROI_EDGE_OBJECTS if args.scope=="roi" else False); cfg.validate()
    images,_=split_images(args.data,args.split); labels={path:read_labels(label_path(path)) for path in images}; detector=create_detector(cfg); rows=[]
    for path in images:
        image=read_image(path); bounds=roi_bounds(image.shape,cfg,args.roi_source); x1,y1,x2,y2=bounds; started=time.perf_counter()
        inference=run_inference(image[y1:y2,x1:x2],detector,cfg)
        predicted=sum(inside_roi_policy(d.bbox,x2-x1,y2-y1,cfg) for d in inference["merged"]); gt=ground_truth_count(labels[path],image.shape,bounds,cfg)
        row={"image":str(path),"ground_truth":gt,"predicted":predicted,"error":predicted-gt,"absolute_error":abs(predicted-gt),"density":density_bucket(gt),"inference_merge_seconds":time.perf_counter()-started}
        rows.append(row); print(f"{path.name}: GT={gt} prediction={predicted} error={predicted-gt:+d}")
    buckets=["0 (negative)","1-20","21-50","51-100","101-200","201+"]
    report={"scope":args.scope,"roi_source":args.roi_source,"split":args.split,"detector":detector.name,"overall":metrics(rows),
            "by_density":{key:metrics([r for r in rows if r["density"]==key]) for key in buckets}}
    out=args.output or cfg.RUNS_DIR/("evaluation_"+unique_id()); out.mkdir(parents=True,exist_ok=False)
    with (out/"per_image.csv").open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/"summary.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8"); print(f"Report: {out}")
if __name__=="__main__": main()
