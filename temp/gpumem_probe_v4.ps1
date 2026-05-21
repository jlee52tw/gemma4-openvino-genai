# GPU memory probe v4 — uses RUN_GEMMA4_PID_FILE to find actual python pid.
param(
    [Parameter(Mandatory=$true)] [string]$Tag,
    [Parameter(Mandatory=$true)] [string]$EnvScript,
    [string]$LogDir = "C:\working\gemma4-openvino-genai\temp"
)
$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

. "C:\working\gemma4-openvino-genai\.venv\Scripts\Activate.ps1"
. $EnvScript

$python  = (Get-Command python).Source
$workdir = "C:\working\gemma4-openvino-genai\gemma4_streaming_release_v2"
$promptFile = "C:\working\gemma4-openvino-genai\temp\prompt_text.txt"
$pidFile    = Join-Path $LogDir "gpumem_$Tag.pid"
if (Test-Path $pidFile) { Remove-Item $pidFile -Force }
$env:RUN_GEMMA4_PID_FILE = $pidFile

$csv     = Join-Path $LogDir "gpumem_$Tag.csv"
$stdout  = Join-Path $LogDir "gpumem_$Tag.stdout.log"
$summary = Join-Path $LogDir "gpumem_$Tag.summary.txt"

Write-Host "[gpumem-v4] launching python, tag=$Tag" -ForegroundColor Cyan
$pyArgs = @("run_gemma4.py","--model-dir","model","--device","GPU",
            "--prompt-file",$promptFile,"--max-new-tokens","256","--show-memory")
$proc = Start-Process -FilePath $python -ArgumentList $pyArgs `
    -WorkingDirectory $workdir `
    -RedirectStandardOutput $stdout -RedirectStandardError "$stdout.err" `
    -PassThru -WindowStyle Hidden
$launcherPid = $proc.Id

# Wait for pid file (Python writes its real os.getpid() there)
$realPid = $launcherPid
$waitStart = Get-Date
while (-not (Test-Path $pidFile)) {
    if ($proc.HasExited) { break }
    if (((Get-Date) - $waitStart).TotalSeconds -gt 30) { break }
    Start-Sleep -Milliseconds 100
}
if (Test-Path $pidFile) {
    try { $realPid = [int](Get-Content $pidFile -Raw).Trim() } catch {}
}
Write-Host "[gpumem-v4] launcher_pid=$launcherPid real_pid=$realPid"

"time_ms,dedicated_MB,local_MB,shared_MB,committed_MB,instances" | Set-Content -Path $csv -Encoding ascii
$start = Get-Date
$samples = 0
$instSearch = "pid_${realPid}_"
$cachedDed = @(); $cachedLoc = @(); $cachedShr = @(); $cachedTot = @()
$lastRefresh = $start.AddSeconds(-10)
while (-not $proc.HasExited) {
    $now = ((Get-Date) - $start).TotalMilliseconds
    try {
        if (((Get-Date) - $lastRefresh).TotalMilliseconds -gt 800) {
            $allPaths = (Get-Counter -ListSet 'GPU Process Memory' -ErrorAction Stop).PathsWithInstances
            $cachedDed = @($allPaths | Where-Object { $_ -like "*($instSearch*" -and $_ -like '*Dedicated Usage' })
            $cachedLoc = @($allPaths | Where-Object { $_ -like "*($instSearch*" -and $_ -like '*Local Usage' -and $_ -notlike '*Non Local*' })
            $cachedShr = @($allPaths | Where-Object { $_ -like "*($instSearch*" -and $_ -like '*Shared Usage' })
            $cachedTot = @($allPaths | Where-Object { $_ -like "*($instSearch*" -and $_ -like '*Total Committed' })
            $lastRefresh = Get-Date
        }
        function SumPaths($paths) {
            if ($paths.Count -eq 0) { return 0 }
            try {
                $c = Get-Counter -Counter $paths -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop
                return ($c.CounterSamples | Measure-Object CookedValue -Sum).Sum
            } catch { return 0 }
        }
        $ded = SumPaths $cachedDed
        $loc = SumPaths $cachedLoc
        $shr = SumPaths $cachedShr
        $tot = SumPaths $cachedTot
        $cnt = $cachedDed.Count
        $line = "{0:F0},{1:F1},{2:F1},{3:F1},{4:F1},{5}" -f $now, ($ded/1MB), ($loc/1MB), ($shr/1MB), ($tot/1MB), $cnt
        Add-Content -Path $csv -Value $line
        $samples++
    } catch {
        Add-Content -Path $csv -Value ("{0:F0},,,,,err" -f $now)
    }
    Start-Sleep -Milliseconds 250
}
$proc.WaitForExit()
$dur = ((Get-Date) - $start).TotalSeconds
Write-Host "[gpumem-v4] python exited code=$($proc.ExitCode) samples=$samples dur=${dur}s" -ForegroundColor Green

$rows = Import-Csv $csv | Where-Object { $_.dedicated_MB -and $_.dedicated_MB -ne '' -and $_.instances -ne '0' } | ForEach-Object {
    [pscustomobject]@{
        time_ms=[double]$_.time_ms; ded=[double]$_.dedicated_MB; loc=[double]$_.local_MB
        shr=[double]$_.shared_MB;   tot=[double]$_.committed_MB; n=[int]$_.instances
    }
}
function Stats($v) {
    $arr = @($v)
    if ($arr.Count -eq 0) { return @{peak=0; median=0; mean=0; final=0} }
    $s = $arr | Sort-Object
    return @{ peak=$s[-1]; median=$s[[int]($s.Count/2)]; mean=($arr|Measure-Object -Average).Average; final=$arr[-1] }
}
$ds=Stats(@($rows|%{$_.ded})); $ls=Stats(@($rows|%{$_.loc})); $ss=Stats(@($rows|%{$_.shr})); $ts=Stats(@($rows|%{$_.tot}))
$out = @"
GPU memory probe v4 - tag=$Tag
samples_with_data=$($rows.Count) total_samples=$samples duration=${dur}s real_pid=$realPid exit=$($proc.ExitCode)

Counter            peak_MB    median_MB    mean_MB    final_MB
Dedicated Usage    {0,8:F1}   {1,8:F1}    {2,8:F1}   {3,8:F1}
Local Usage        {4,8:F1}   {5,8:F1}    {6,8:F1}   {7,8:F1}
Shared Usage       {8,8:F1}   {9,8:F1}    {10,8:F1}  {11,8:F1}
Total Committed   {12,8:F1}  {13,8:F1}   {14,8:F1}  {15,8:F1}
"@ -f $ds.peak,$ds.median,$ds.mean,$ds.final,
       $ls.peak,$ls.median,$ls.mean,$ls.final,
       $ss.peak,$ss.median,$ss.mean,$ss.final,
       $ts.peak,$ts.median,$ts.mean,$ts.final
$out | Tee-Object -FilePath $summary
