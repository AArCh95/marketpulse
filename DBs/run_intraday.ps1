# D:\AARCH\DBs\run_intraday.ps1
$ErrorActionPreference = "Stop"
$logDir = "D:\AARCH\DBs\logs"
New-Item -Force -ItemType Directory -Path $logDir | Out-Null

# Env for API to find snapshots
$env:DB_SNAPSHOT_POINTER = "D:\AARCH\DBs\CURRENT_SNAPSHOT.txt"

# Absolute Python path (adjust if different)
$python = "C:\Users\Aaron\AppData\Local\Programs\Python\Python313\python.exe"
$script = "D:\AARCH\DBs\market_sync_yahoo.py"
$db     = "D:\AARCH\DBs\market.duckdb"
$log    = Join-Path $logDir ("intraday_" + (Get-Date).ToString("yyyyMMdd_HHmmss") + ".log")

# INTRADAY mode (blocks until you stop the Task)
& $python $script --duckdb $db --mode intraday --batch-size 60 --sleep 0.35 --snapshot-dir "D:\AARCH\DBs" *>$log
