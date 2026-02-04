#!/usr/bin/env python3
# D:\AARCH\models\catboost_intraday\validate_setup.py
# Validation script to ensure everything is configured correctly

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def check_file_exists(path: Path, name: str) -> bool:
    """Check if a required file exists."""
    if path.exists():
        print(f"  ✓ {name}: {path}")
        return True
    else:
        print(f"  ✗ {name}: NOT FOUND at {path}")
        return False

def check_dependencies():
    """Check if all required packages are installed."""
    print("\n[1/5] Checking dependencies...")
    
    required = [
        ("catboost", "CatBoost"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("duckdb", "DuckDB"),
        ("sklearn", "scikit-learn"),
        ("dateutil", "python-dateutil"),
    ]
    
    missing = []
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n  Install missing packages:")
        print(f"    pip install {' '.join(missing.lower())}")
        return False
    
    return True

def check_files():
    """Check if all required files exist."""
    print("\n[2/5] Checking files...")
    
    root = Path(r"D:\AARCH\models\catboost_intraday")
    
    required_files = [
        (root / "config_intraday.py", "Config"),
        (root / "db_intraday.py", "DB module"),
        (root / "features_intraday.py", "Features module"),
        (root / "train_intraday.py", "Training script"),
        (root / "serve_intraday.py", "API server"),
        (root / "retrain_weekly.py", "Retrain script"),
    ]
    
    all_exist = True
    for path, name in required_files:
        if not check_file_exists(path, name):
            all_exist = False
    
    return all_exist

def check_database():
    """Check if DuckDB exists and has required tables."""
    print("\n[3/5] Checking database...")
    
    try:
        import duckdb
        
        # Check snapshot pointer
        snapshot_pointer = Path(r"D:\AARCH\DBs\CURRENT_SNAPSHOT.txt")
        
        if snapshot_pointer.exists():
            try:
                snapshot_path = snapshot_pointer.read_text(encoding="utf-8").strip().strip('"').strip("'")
                print(f"  ✓ Snapshot pointer exists: {snapshot_pointer}")
                print(f"    Points to: {snapshot_path}")
                
                if Path(snapshot_path).exists():
                    print(f"  ✓ Snapshot file exists: {snapshot_path}")
                    db_path = snapshot_path
                else:
                    print(f"  ✗ Snapshot file NOT FOUND: {snapshot_path}")
                    print(f"    Falling back to primary DB")
                    db_path = r"D:\AARCH\DBs\market.duckdb"
            except Exception as e:
                print(f"  ⚠ Failed to read snapshot pointer: {e}")
                db_path = r"D:\AARCH\DBs\market.duckdb"
        else:
            print(f"  ⚠ Snapshot pointer not found: {snapshot_pointer}")
            print(f"    Using primary DB (may cause lock issues during training)")
            db_path = r"D:\AARCH\DBs\market.duckdb"
        
        if not Path(db_path).exists():
            print(f"  ✗ Database not found: {db_path}")
            return False
        
        print(f"  ✓ Database exists: {db_path}")
        
        # Check tables and row counts
        conn = duckdb.connect(db_path, read_only=True)
        
        # Check ohlcv_1m
        count_1m = conn.execute("SELECT COUNT(*) FROM ohlcv_1m").fetchone()[0]
        max_ts_1m = conn.execute("SELECT MAX(ts) FROM ohlcv_1m").fetchone()[0]
        symbols_1m = conn.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv_1m").fetchone()[0]
        
        print(f"  ✓ ohlcv_1m: {count_1m:,} rows, {symbols_1m} symbols")
        print(f"    Latest timestamp: {max_ts_1m}")
        
        # Check if data is recent (within 7 days)
        if max_ts_1m:
            try:
                # Parse timestamp and make timezone-aware for comparison
                latest_str = str(max_ts_1m).replace("+00:00", "").replace("Z", "")
                if "-06:00" in latest_str or "+00:00" in latest_str:
                    # Already has timezone info, parse as-is
                    latest = pd.to_datetime(latest_str)
                else:
                    # Assume UTC if no timezone
                    latest = pd.to_datetime(latest_str, utc=True)
                
                # Make now() timezone-aware for comparison
                now = pd.Timestamp.now(tz="UTC")
                
                # Convert latest to UTC if needed
                if latest.tz is None:
                    latest = latest.tz_localize("UTC")
                else:
                    latest = latest.tz_convert("UTC")
                
                age_days = (now - latest).days
                
                if age_days > 7:
                    print(f"  ⚠ WARNING: Data is {age_days} days old. Run market_sync_yahoo.py")
                else:
                    print(f"  ✓ Data freshness: {age_days} days old")
            except Exception as e:
                print(f"  ⚠ Could not check data freshness: {e}")
                print(f"    Latest timestamp: {max_ts_1m}")
        
        # Check if dist_vwap_bps exists and has non-null values
        vwap_count = conn.execute("""
            SELECT COUNT(*) 
            FROM ohlcv_1m 
            WHERE dist_vwap_bps IS NOT NULL
            LIMIT 1000
        """).fetchone()[0]
        
        if vwap_count > 0:
            print(f"  ✓ dist_vwap_bps: populated")
        else:
            print(f"  ⚠ WARNING: dist_vwap_bps is NULL. Run update_features.py")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Database check failed: {e}")
        return False

def check_training_feasibility():
    """Estimate training data size."""
    print("\n[4/5] Checking training feasibility...")
    
    try:
        import duckdb
        from config_intraday import TRAIN_DAYS, TRAIN_SYMBOLS_LIMIT, RTH_START, RTH_END
        
        db_path = r"D:\AARCH\DBs\market.duckdb"
        conn = duckdb.connect(db_path, read_only=True)
        
        # Count bars that would be used for training
        q = f"""
            WITH recent AS (
                SELECT COUNT(*) AS bar_count
                FROM ohlcv_1m
                WHERE ts >= CURRENT_DATE - INTERVAL '{TRAIN_DAYS} days'
                  AND EXTRACT(HOUR FROM ts AT TIME ZONE 'America/New_York') * 60 
                      + EXTRACT(MINUTE FROM ts AT TIME ZONE 'America/New_York') 
                      BETWEEN 
                          (CAST(SPLIT_PART('{RTH_START}', ':', 1) AS INT) * 60 + CAST(SPLIT_PART('{RTH_START}', ':', 2) AS INT))
                      AND 
                          (CAST(SPLIT_PART('{RTH_END}', ':', 1) AS INT) * 60 + CAST(SPLIT_PART('{RTH_END}', ':', 2) AS INT))
            )
            SELECT bar_count FROM recent
        """
        
        total_bars = conn.execute(q).fetchone()[0]
        
        print(f"  Training window: {TRAIN_DAYS} days, {TRAIN_SYMBOLS_LIMIT} symbols (max)")
        print(f"  RTH filter: {RTH_START} - {RTH_END} ET")
        print(f"  Total bars available: {total_bars:,}")
        
        # Estimate
        estimated_per_symbol = total_bars // TRAIN_SYMBOLS_LIMIT if TRAIN_SYMBOLS_LIMIT else 0
        print(f"  Estimated bars per symbol: {estimated_per_symbol:,}")
        
        if total_bars < 10000:
            print(f"  ✗ INSUFFICIENT DATA: Need at least 10,000 bars")
            print(f"    Current: {total_bars:,}")
            print(f"    Run market_sync_yahoo.py to collect more data")
            return False
        
        print(f"  ✓ Sufficient data for training")
        
        # Estimate training time (rough: 500 bars/sec on CPU)
        est_minutes = total_bars / 500 / 60
        print(f"  Estimated training time: {est_minutes:.0f}-{est_minutes*2:.0f} minutes")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Training feasibility check failed: {e}")
        return False

def check_model():
    """Check if model is trained and loadable."""
    print("\n[5/5] Checking model...")
    
    try:
        from config_intraday import MODEL_PATH, FEATURE_ORDER_PATH
        from catboost import CatBoostClassifier
        
        if not MODEL_PATH.exists():
            print(f"  ⚠ Model not trained yet: {MODEL_PATH}")
            print(f"    Run: python train_intraday.py")
            return False
        
        print(f"  ✓ Model file exists: {MODEL_PATH}")
        
        # Try loading
        model = CatBoostClassifier()
        model.load_model(str(MODEL_PATH))
        
        print(f"  ✓ Model loads successfully")
        print(f"    Features: {model.feature_count_}")
        print(f"    Trees: {model.tree_count_}")
        print(f"    Classes: {model.classes_count_}")
        
        # Check feature order
        if FEATURE_ORDER_PATH.exists():
            print(f"  ✓ Feature order file exists")
        else:
            print(f"  ⚠ Feature order file missing (non-critical)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model check failed: {e}")
        return False

def main():
    """Run all validation checks."""
    print("="*70)
    print("CATBOOST INTRADAY 5M - SETUP VALIDATION")
    print("="*70)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Files", check_files),
        ("Database", check_database),
        ("Training Feasibility", check_training_feasibility),
        ("Model", check_model),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8s} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. python train_intraday.py")
        print("  2. python serve_intraday.py")
        print("  3. Update n8n workflow to call http://localhost:8001/score_batch")
    else:
        print("\n✗ Some checks failed. Fix issues above before training.")
        
        # Specific guidance
        if not results["Database"]:
            print("\nDatabase issues:")
            print("  - Ensure market_sync_yahoo.py is running")
            print("  - Check D:\\AARCH\\DBs\\market.duckdb exists")
            print("  - Run update_features.py if dist_vwap_bps is NULL")
        
        if not results["Files"]:
            print("\nFile issues:")
            print("  - Copy all Python files to D:\\AARCH\\models\\catboost_intraday\\")
        
        if not results["Dependencies"]:
            print("\nDependency issues:")
            print("  - Run: pip install -r requirements.txt")
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
