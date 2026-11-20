# Post-training pipeline for Seamless APMGSRN. Run from the APMGSRN folder after the
# three training configs have finished:
#   Code/start_jobs.py --settings train_plume_dd_g1.json  (and _g4, _g16)
param(
    [string]$Python = "D:\Computer_Graphics_env\.venv\Scripts\python.exe",
    [string]$Device = "cuda:0",
    [int]$FinetuneIterations = 1500,
    [double]$ValueWeight = 1.0,
    [double]$GradientWeight = 0.05,
    [switch]$SkipFinetune,
    [switch]$SkipRender
)
$ErrorActionPreference = "Stop"
$out = "Output/SeamlessAPMGSRN"
New-Item -ItemType Directory -Force $out | Out-Null
$timing = [ordered]@{}

function Timed($name, [scriptblock]$block) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE" }
    $timing[$name] = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    Write-Host "$name took $($timing[$name]) s"
}

if (-not $SkipFinetune) {
    Timed "finetune_g4" { & $Python Code/finetune_boundaries.py --load_from Plume_APMGSRN_dd_g4 `
        --save_name Plume_APMGSRN_dd_g4_consistency --device $Device --data_device $Device `
        --iterations $FinetuneIterations --points_per_iteration 4096 --interior_points 2048 `
        --lr 0.0005 --value_weight $ValueWeight --gradient_weight $GradientWeight `
        --warmup_iterations 250 --seed 42 --max_minutes 15 --overwrite true }
}

$runs = @(
    @("g1_hard",              "Plume_APMGSRN_dd_g1",             "hard"),
    @("g4_hard",              "Plume_APMGSRN_dd_g4",             "hard"),
    @("g4_blend",             "Plume_APMGSRN_dd_g4",             "blend"),
    @("g4_consistency_hard",  "Plume_APMGSRN_dd_g4_consistency", "hard"),
    @("g4_consistency_blend", "Plume_APMGSRN_dd_g4_consistency", "blend"),
    @("g16_hard",             "Plume_APMGSRN_dd_g16",            "hard"),
    @("g16_blend",            "Plume_APMGSRN_dd_g16",            "blend")
)
foreach ($r in $runs) {
    if (-not (Test-Path "SavedModels/$($r[1])")) { Write-Host "Skipping $($r[0]): $($r[1]) missing"; continue }
    Timed "eval_$($r[0])" { & $Python Code/evaluate_seams.py --load_from $r[1] --routing_mode $r[2] `
        --run_id $r[0] --device $Device --data_device cpu --boundary_band_cells 2 `
        --face_samples 65536 --throughput_points 262144 --throughput_repeats 20 --seed 42 }
}

if (-not $SkipRender) {
    $specs = @()
    foreach ($r in $runs) { if (Test-Path "SavedModels/$($r[1])") { $specs += "$($r[0])=$($r[1]):$($r[2])" } }
    Timed "render" { & $Python Code/compare_renderings.py --runs @specs --device $Device --hw 512,256 --samples 768 }
}

Timed "summarize" { & $Python Code/summarize_seamless.py }
$timing | ConvertTo-Json | Set-Content -Encoding utf8NoBOM "$out/timing.json"
