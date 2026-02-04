#!/usr/bin/env python3
# D:\AARCH\models\catboost_intraday\retrain_weekly.py
# Automated weekly retraining script for 5-minute model

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import shutil

# Directories
PROJECT_ROOT = Path(r"D:\AARCH\models\catboost_intraday")
BACKUP_DIR = PROJECT_ROOT / "model_backups"
TRAIN_SCRIPT = PROJECT_ROOT / "train_intraday.py"

def backup_current_model():
    """Backup current model before retraining."""
    from config_intraday import MODEL_PATH, FEATURE_ORDER_PATH
    
    if not MODEL_PATH.exists():
        print("[BACKUP] No existing model to backup.")
        return
    
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_model = BACKUP_DIR / f"catboost_intraday_5m_{timestamp}.cbm"
    backup_features = BACKUP_DIR / f"feature_order_intraday_{timestamp}.json"
    
    shutil.copy2(MODEL_PATH, backup_model)
    print(f"[BACKUP] Model saved to {backup_model}")
    
    if FEATURE_ORDER_PATH.exists():
        shutil.copy2(FEATURE_ORDER_PATH, backup_features)
        print(f"[BACKUP] Feature order saved to {backup_features}")


def run_training():
    """Execute training script."""
    print(f"\n{'='*60}")
    print(f"Starting training: {datetime.now()}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n[ERROR] Training failed with exit code {result.returncode}")
        return False
    
    print(f"\n[SUCCESS] Training completed at {datetime.now()}")
    return True


def validate_new_model():
    """Basic validation that new model exists and loads."""
    from config_intraday import MODEL_PATH
    from catboost import CatBoostClassifier
    
    if not MODEL_PATH.exists():
        print("[ERROR] New model file not found.")
        return False
    
    try:
        model = CatBoostClassifier()
        model.load_model(str(MODEL_PATH))
        print(f"[VALIDATE] Model loaded successfully.")
        print(f"  Feature count: {model.feature_count_}")
        print(f"  Tree count: {model.tree_count_}")
        return True
    except Exception as e:
        print(f"[ERROR] Model validation failed: {e}")
        return False


def main():
    """Main retraining workflow."""
    print(f"\n{'='*70}")
    print(f"CATBOOST INTRADAY 5M - WEEKLY RETRAIN")
    print(f"Started: {datetime.now()}")
    print(f"{'='*70}\n")
    
    # Step 1: Backup current model
    print("[1/4] Backing up current model...")
    backup_current_model()
    
    # Step 2: Train new model
    print("\n[2/4] Training new model...")
    if not run_training():
        print("\n[ABORT] Retraining failed. Exiting.")
        sys.exit(1)
    
    # Step 3: Validate new model
    print("\n[3/4] Validating new model...")
    if not validate_new_model():
        print("\n[ABORT] Model validation failed. Exiting.")
        sys.exit(1)
    
    # Step 4: Cleanup old backups (keep last 10)
    print("\n[4/4] Cleaning up old backups...")
    backups = sorted(BACKUP_DIR.glob("catboost_intraday_5m_*.cbm"))
    if len(backups) > 10:
        for old in backups[:-10]:
            old.unlink()
            # Also delete corresponding feature order file
            feat_file = old.with_name(old.stem.replace("catboost_intraday_5m", "feature_order_intraday") + ".json")
            if feat_file.exists():
                feat_file.unlink()
            print(f"  Deleted: {old.name}")
    
    print(f"\n{'='*70}")
    print(f"RETRAIN COMPLETE")
    print(f"Finished: {datetime.now()}")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Restart serve_intraday.py to load new model")
    print(f"  2. Monitor predictions for anomalies")
    print(f"  3. Check model_backups/ for previous versions")


if __name__ == "__main__":
    main()
