from pathlib import Path
import cv2
import numpy as np
import yaml
IMAGE_SUFFIXES = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def read_image(path):
    image=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f"Cannot decode {path}")
    return image

def load_dataset(yaml_path):
    yaml_path=Path(yaml_path).resolve(); data=yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data,dict) or data.get("names") not in ({0:"object"},["object"]):
        raise ValueError("Dataset names must be exactly {0: object}")
    root=Path(data.get("path","."))
    if not root.is_absolute(): root=(yaml_path.parent/root).resolve()
    return data,root

def split_images(yaml_path,split):
    data,root=load_dataset(yaml_path); entry=data.get(split)
    if not isinstance(entry,str): raise ValueError(f"{split} must name an images directory")
    directory=Path(entry); directory=root/directory if not directory.is_absolute() else directory
    if not directory.is_dir() or "images" not in directory.parts: raise ValueError(f"Missing images directory: {directory}")
    images=sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images: raise ValueError(f"No images in {directory}; collect and annotate your dataset first")
    return images,directory

def label_path(image):
    parts=list(Path(image).parts); index=len(parts)-1-parts[::-1].index("images"); parts[index]="labels"
    return Path(*parts).with_suffix(".txt")

def read_labels(path):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(f"Missing annotation {path}; negatives require an empty .txt")
    boxes=[]
    for number,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        parts=line.split()
        if len(parts)!=5: raise ValueError(f"{path}:{number}: expected class cx cy w h (5 columns)")
        values=np.array([float(x) for x in parts],dtype=float); cls,cx,cy,w,h=values
        if not np.isfinite(values).all() or cls!=0 or w<=0 or h<=0: raise ValueError(f"{path}:{number}: invalid class or box")
        box=(cx-w/2,cy-h/2,cx+w/2,cy+h/2)
        if min(box)<-1e-5 or max(box)>1+1e-5: raise ValueError(f"{path}:{number}: box lies outside normalized image")
        boxes.append(tuple(float(v) for v in np.clip(box,0,1)))
    return boxes

def validate_dataset(yaml_path,splits=("train","val")):
    resolved,root=load_dataset(yaml_path); resolved=dict(resolved,path=str(root)); report={}
    for split in splits:
        images,directory=split_images(yaml_path,split); total=sum(len(read_labels(label_path(image))) for image in images)
        if split=="train" and not total: raise ValueError("Training split has no annotated objects")
        report[split]={"images":len(images),"objects":total}; resolved[split]=str(directory.resolve())
    return resolved,report
