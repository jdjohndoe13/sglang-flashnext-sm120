# watch_server_leaks.ps1 — server-side leak watcher for the sglang box (testcomp2)
#
# Tails logs/serve.log on the server via ssh and prints an alert the moment
# NEW server-side leak-signal lines appear:
#   "Tool 'X' is not defined in the tools list"        -> unknown-tool class (strong leak signal)
#   "Parsed parameter 'X' is not defined in the tool parameters" -> bad-param class (model-internal,
#       also present on clean boots; informational, not a proven leak)
#
# Baselines the current counts on start, so pre-existing hits from the
# current boot are NOT re-alerted. Polls every 30s. Ctrl+C to stop.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File scripts\bench\watch_server_leaks.ps1

$ErrorActionPreference = 'Continue'
$remote = 'testcomp2'
$log = '/mnt/data/shared/models/sglang-flashnext-sm120/logs/serve.log'
$patTools = 'not defined in the tools list'
$patParams = 'not defined in the tool parameters'

function RemoteCount([string]$pattern) {
  $cmd = "grep -c -F '$pattern' $log"
  $out = ssh -o ConnectTimeout=10 -o BatchMode=yes $remote $cmd 2>$null
  if ($LASTEXITCODE -eq 2) { return -1 }   # ssh/grep error
  if ($out -match '^\d+$') { return [int]$out }
  return 0
}

$baseTools = RemoteCount $patTools
$baseParams = RemoteCount $patParams
if ($baseTools -lt 0) { Write-Host '!! cannot reach testcomp2 over ssh'; exit 2 }
Write-Host "watching $remote : $log"
Write-Host "baseline: tools-list=$baseTools  bad-params=$baseParams  (pre-existing, not alerted)"
Write-Host "polling every 30s - Ctrl+C to stop"

$lastTools = $baseTools
$lastParams = $baseParams
while ($true) {
  Start-Sleep -Seconds 30
  $t = RemoteCount $patTools
  $p = RemoteCount $patParams
  if ($t -lt 0) { Write-Host "[!] ssh check failed at $(Get-Date -Format HH:mm:ss)"; continue }
  # count dropped below what we last saw -> the log was truncated (server
  # restart). Re-baseline so the restart's own hits are still detected.
  if ($t -lt $lastTools) { Write-Host "[i] serve.log truncated (restart?) at $(Get-Date -Format HH:mm:ss) - re-baselining tools-list=$t" -ForegroundColor Cyan; $lastTools = $t }
  if ($p -lt $lastParams) { $lastParams = $p }
  if ($t -gt $lastTools) {
    $new = $t - $lastTools
    Write-Host ("!!! SERVER-SIDE UNKNOWN-TOOL: {0} new line(s) (total {1} vs baseline {2})  {3}" -f $new, $t, $baseTools, (Get-Date -Format 'HH:mm:ss')) -ForegroundColor Red
    # show the newest occurrences for context
    $lastTools = $t
  }
  if ($p -gt $lastParams) {
    $new = $p - $lastParams
    Write-Host ("[i] server-side bad-param lines: {0} new (total {1}) - model-internal class, informational" -f $new, $p) -ForegroundColor Yellow
    $lastParams = $p
  }
}
