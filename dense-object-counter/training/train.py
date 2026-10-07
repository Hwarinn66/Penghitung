"""Fine-tune a pretrained YOLO detection model, then install its best weights."""
import argparse
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from config import settings
from counting.counter import unique_id
from training.dataset_utils import validate_dataset

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.DATASET_YAML)
    parser.add_argument("--model", default=settings.MODEL_NAME)
    parser.add_argument("--imgsz", type=int, default=settings.IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS)
    parser.add_argument("--device", default=settings.DEVICE)
    parser.add_argument("--workers", type=int, default=settings.TRAIN_WORKERS)
    args = parser.parse_args()
    if Path(args.model).suffix != ".pt":
        parser.error("Use pretrained .pt weights, e.g. yolo11n.pt; .yaml training from scratch is disabled")
    if args.imgsz < 32 or args.imgsz % 32 or args.batch < 1 or args.epochs < 1:
        parser.error("imgsz must be a positive multiple of 32; batch and epochs must be positive")
    resolved, report = validate_dataset(args.data)
    run_dir = settings.RUNS_DIR / ("train_" + unique_id())
    run_dir.mkdir(parents=True)
    data_path = run_dir / "dataset.resolved.yaml"
    data_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    print("Dataset:", report)
    from ultralytics import YOLO
    model = YOLO(args.model)
    if model.task != "detect":
        parser.error("This MVP trains bounding-box detection; use a detect pretrained model")
    params = dict(data=str(data_path), imgsz=args.imgsz, batch=args.batch, epochs=args.epochs,
                  workers=args.workers, single_cls=True, pretrained=True, seed=settings.TRAIN_SEED,
                  max_det=settings.MAX_DETECTIONS_PER_TILE, project=str(run_dir), name="fit", exist_ok=False)
    if args.device: params["device"] = args.device
    model.train(**params)
    best = Path(model.trainer.best)
    if not best.is_file(): raise RuntimeError("Training did not produce best.pt")
    target = settings.MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): shutil.copy2(target, target.with_name(f"best_backup_{unique_id()}.pt"))
    shutil.copy2(best, target)
    print(f"Best model installed: {target}")
if __name__ == "__main__":
    main()
