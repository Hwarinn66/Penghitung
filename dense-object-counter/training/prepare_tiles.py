"""Tile train/val AFTER session-based split. Test images remain full-frame."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import yaml
from config import settings
from counting.counter import save_image
from detection.tiled_inference import make_tiles
from training.dataset_utils import label_path,read_image,read_labels,split_images

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=settings.DATASET_YAML); parser.add_argument("--output",type=Path,default=ROOT/"dataset_tiles")
    parser.add_argument("--tile-size",type=int,default=settings.TILE_SIZE); parser.add_argument("--overlap",type=float,default=settings.TILE_OVERLAP)
    parser.add_argument("--min-visible",type=float,default=settings.TRAIN_TILE_MIN_VISIBLE)
    args=parser.parse_args()
    if not 0<=args.min_visible<=1: parser.error("min-visible must be in [0, 1]")
    destination=args.output.resolve()
    if destination.exists(): parser.error("Output already exists; choose a new directory")
    destination.mkdir(parents=True); summary={}
    for split in ("train","val"):
        images,_=split_images(args.data,split); image_dir,label_dir=destination/"images"/split,destination/"labels"/split
        image_dir.mkdir(parents=True); label_dir.mkdir(parents=True); written=skipped=0
        for image_index,image_path in enumerate(images):
            image=read_image(image_path); h,w=image.shape[:2]
            boxes=[(a*w,b*h,c*w,d*h) for a,b,c,d in read_labels(label_path(image_path))]
            for tile in make_tiles(w,h,args.tile_size,args.overlap):
                tw,th=tile.x2-tile.x1,tile.y2-tile.y1; lines=[]; reject=False
                for x1,y1,x2,y2 in boxes:
                    a,b,c,d=max(x1,tile.x1),max(y1,tile.y1),min(x2,tile.x2),min(y2,tile.y2)
                    if c<=a or d<=b: continue
                    if (c-a)*(d-b)/((x2-x1)*(y2-y1))<args.min_visible or min(c-a,d-b)<1: reject=True; break
                    lines.append(f"0 {((a+c)/2-tile.x1)/tw:.8f} {((b+d)/2-tile.y1)/th:.8f} {(c-a)/tw:.8f} {(d-b)/th:.8f}")
                if reject: skipped+=1; continue
                name=f"frame_{image_index:06d}_tile_{tile.id:04d}"
                save_image(image_dir/f"{name}.jpg",image[tile.y1:tile.y2,tile.x1:tile.x2],98)
                (label_dir/f"{name}.txt").write_text("\n".join(lines),encoding="utf-8"); written+=1
        if not written: raise ValueError(f"No {split} tiles produced")
        summary[split]={"written":written,"skipped":skipped,"sources":[str(p) for p in images]}
    _,test_dir=split_images(args.data,"test")
    data={"path":str(destination),"train":"images/train","val":"images/val","test":str(test_dir.resolve()),"names":{0:"object"}}
    (destination/"dataset.yaml").write_text(yaml.safe_dump(data,sort_keys=False),encoding="utf-8")
    (destination/"manifest.yaml").write_text(yaml.safe_dump(summary,sort_keys=False),encoding="utf-8")
    print("Tile dataset:",destination/"dataset.yaml")
if __name__=="__main__": main()
