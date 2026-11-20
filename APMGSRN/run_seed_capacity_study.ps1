# Seed + capacity study for Seamless APMGSRN. Resumable: every step is skipped if its output exists.
# Run from the APMGSRN folder. Training runs strictly one process at a time (page-file limit).
param(
    [string]$Python = "D:\Computer_Graphics_env\.venv\Scripts\python.exe",
    [string]$Device = "cuda:0",
    [string]$OutRoot = "Output/SeedCapacityStudy",
    [switch]$SkipTraining,
    [switch]$SkipRender,
    [string]$TrainDeadline = ""     # HH:mm; no new training/fine-tuning starts after this
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutRoot, "Code/BatchRunSettings/study" | Out-Null
$log = "$OutRoot/study_progress.log"
function Note($m) { $line = "$(Get-Date -Format 'HH:mm:ss') $m"; Write-Host $line; Add-Content $log $line }

# ghost = actual overlap cells, feat = feature grid edge, seeds, finetune = run our consistency fine-tune.
# Ordered by priority so the key comparisons finish first if the GPU budget runs out.
$conditions = @(
    @{ ghost = 4;  feat = 24; seeds = @(42, 1, 2); finetune = $true  },
    @{ ghost = 16; feat = 24; seeds = @(42, 1, 2); finetune = $false },
    @{ ghost = 16; feat = 27; seeds = @(42);       finetune = $false },   # density-matched g16 (+42% params)
    @{ ghost = 4;  feat = 24; seeds = @(2, 42);    finetune = $false; control = $true },  # recon-only fine-tune control
    @{ ghost = 16; feat = 24; seeds = @(42);       finetune = $false; control = $true },   # same extra training for g16
    @{ ghost = 16; feat = 32; seeds = @(42);       finetune = $false },   # g16 with +135% params
    @{ ghost = 16; feat = 27; seeds = @(1);        finetune = $false },
    @{ ghost = 4;  feat = 24; seeds = @(1);        finetune = $false; control = $true },
    @{ ghost = 1;  feat = 24; seeds = @(42, 1, 2); finetune = $false },
    @{ ghost = 2;  feat = 24; seeds = @(42);       finetune = $true  },
    @{ ghost = 4;  feat = 16; seeds = @(42);       finetune = $true  },
    @{ ghost = 16; feat = 16; seeds = @(42);       finetune = $false }
)

function PastDeadline { if (-not $TrainDeadline) { return $false }; return (Get-Date) -gt [datetime]::ParseExact($TrainDeadline, "HH:mm", $null) }
function Complete($n) { return @(Get-ChildItem "SavedModels/$n" -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "model.ckpt.tar") }).Count -eq 8 }

function SaveName($c, $seed) {
    # Seed-42 / 24^3 checkpoints from run 1 keep their original names.
    if ($c.feat -eq 24 -and $seed -eq 42 -and $c.ghost -ne 2) { return "Plume_APMGSRN_dd_g$($c.ghost)" }
    return "Plume_APMGSRN_dd_g$($c.ghost)_f$($c.feat)_s$seed"
}

$specRows = @()
foreach ($c in $conditions) {
    foreach ($seed in $c.seeds) {
        $name = SaveName $c $seed
        $cond = "g$($c.ghost)_f$($c.feat)_s$seed"
        if (-not $SkipTraining) {
            $cfgPath = "Code/BatchRunSettings/study/$name.json"
            $cfg = @"
[
  ["train.py", {
    "save_name": "$name", "model": "APMGSRN", "ensemble": true, "ensemble_grid": "2,2,2",
    "ensemble_ghost_cells": $($c.ghost - 1), "data": "Plume.nc", "n_features": 1, "n_grids": 8,
    "feature_grid_shape": "$($c.feat),$($c.feat),$($c.feat)", "n_layers": 2, "nodes_per_layer": 64,
    "precondition": false, "use_tcnn_if_available": false, "log_every": 0, "log_features_every": 0,
    "lr": 0.01, "points_per_iteration": 32768, "iterations": 5000, "max_training_minutes": 6.0,
    "seed": $seed
  }]
]
"@
            Set-Content -Encoding utf8NoBOM $cfgPath $cfg
            $missing = @(Get-ChildItem "SavedModels/$name" -Directory -ErrorAction SilentlyContinue |
                Where-Object { Test-Path (Join-Path $_.FullName "model.ckpt.tar") }).Count -lt 8
            if ($missing -and (PastDeadline)) { Note "deadline: skip training $name" }
            elseif ($missing) {
                Note "train $name"
                $sw = [Diagnostics.Stopwatch]::StartNew()
                & $Python Code/start_jobs.py --settings "study/$name.json" --devices $Device --data_devices same --skip_existing true | Out-Null
                Note "train $name done in $([math]::Round($sw.Elapsed.TotalMinutes,2)) min"
            }
            if ($c.finetune -and (Complete $name) -and -not (PastDeadline) -and -not (Test-Path "SavedModels/${name}_consistency/boundary_finetune_log.json")) {
                Note "finetune $name"
                & $Python Code/finetune_boundaries.py --load_from $name --save_name "${name}_consistency" `
                    --device $Device --data_device $Device --iterations 1500 --points_per_iteration 4096 `
                    --interior_points 2048 --lr 0.0005 --value_weight 1.0 --gradient_weight 0.05 `
                    --warmup_iterations 250 --seed $seed --max_minutes 15 --overwrite true | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "finetune $name failed" }
                Note "finetune $name done"
            }
        }
        if ($c.control -and -not $SkipTraining -and (Complete $name) -and -not (PastDeadline) -and
            -not (Test-Path "SavedModels/${name}_reconft/boundary_finetune_log.json")) {
            Note "control finetune (reconstruction only) $name"
            & $Python Code/finetune_boundaries.py --load_from $name --save_name "${name}_reconft" `
                --device $Device --data_device $Device --iterations 1500 --points_per_iteration 4096 `
                --interior_points 2048 --lr 0.0005 --value_weight 0.0 --gradient_weight 0.0 `
                --warmup_iterations 250 --seed $seed --max_minutes 15 --overwrite true | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "control finetune $name failed" }
            Note "control finetune $name done"
        }        $variants = @(@("hard", $name, "hard"), @("blend", $name, "blend"))
        if ($c.control) { $variants = @(@("rft_hard", "${name}_reconft", "hard"), @("rft_blend", "${name}_reconft", "blend")) }
        if ($c.finetune) { $variants += ,@("ft_hard", "${name}_consistency", "hard"); $variants += ,@("ft_blend", "${name}_consistency", "blend") }
        foreach ($v in $variants) {
            if (-not (Test-Path "SavedModels/$($v[1])/options.json")) { continue }
            if (($v[1].EndsWith("_consistency") -or $v[1].EndsWith("_reconft")) -and -not (Test-Path "SavedModels/$($v[1])/boundary_finetune_log.json")) { continue }
            if (-not ($v[1].EndsWith("_consistency") -or $v[1].EndsWith("_reconft")) -and -not (Complete $v[1])) { continue }
            $runId = "${cond}_$($v[0])"
            $specRows += [pscustomobject]@{ run_id = $runId; ghost = $c.ghost; feat = $c.feat; seed = $seed;
                variant = $v[0]; checkpoint = $v[1]; mode = $v[2] }
            if (-not (Test-Path "$OutRoot/runs/$runId/metrics.json")) {
                Note "evaluate $runId"
                & $Python Code/evaluate_seams.py --load_from $v[1] --routing_mode $v[2] --run_id $runId `
                    --device $Device --data_device cpu --boundary_band_cells 2 --face_samples 65536 `
                    --throughput_points 262144 --throughput_repeats 20 --seed 42 --output_root $OutRoot | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "evaluate $runId failed" }
            }
        }
        $specRows | Export-Csv -NoTypeInformation "$OutRoot/study_runs.csv"
    }
}

if (-not $SkipRender) {
    Note "render all runs"
    $specs = $specRows | ForEach-Object { "$($_.run_id)=$($_.checkpoint):$($_.mode)" }
    $montage = "g1_f24_s42_hard,g4_f24_s42_hard,g4_f24_s42_blend,g4_f24_s42_ft_blend,g16_f24_s42_hard,g16_f27_s42_hard,g16_f32_s42_hard"
    & $Python Code/compare_renderings.py --runs @specs --device $Device --hw 512,256 --samples 768 `
        --output_root $OutRoot --montage_runs $montage | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "render failed" }
}
Note "aggregate"
& $Python Code/aggregate_study.py --output_root $OutRoot
Note "STUDY DONE"
