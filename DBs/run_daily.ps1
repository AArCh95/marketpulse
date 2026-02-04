# D:\AARCH\DBs\run_daily.ps1
$ErrorActionPreference = "Stop"
$logDir = "D:\AARCH\DBs\logs"
New-Item -Force -ItemType Directory -Path $logDir | Out-Null

# Env for API to find snapshots
$env:DB_SNAPSHOT_POINTER = "D:\AARCH\DBs\CURRENT_SNAPSHOT.txt"

# Absolute Python path (adjust if different)
$python = "C:\Users\Aaron\AppData\Local\Programs\Python\Python313\python.exe"
$script = "D:\AARCH\DBs\market_sync_yahoo.py"
$db     = "D:\AARCH\DBs\market.duckdb"
$log    = Join-Path $logDir ("daily_" + (Get-Date).ToString("yyyyMMdd_HHmmss") + ".log")

# DAILY mode (one-shot)
& $python $script --duckdb $db --mode daily --snapshot-dir "D:\AARCH\DBs" *>$log
