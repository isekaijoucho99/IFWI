@echo off
setlocal
set "DEVICE=%~1"
if "%DEVICE%"=="" set "DEVICE=cuda:0"
set "UPDATES=%~2"
if "%UPDATES%"=="" set "UPDATES=4000"
set "SCRIPT_DIR=%~dp0"
set "OUTPUT_BASE=%~dp0..\results\batch_%RANDOM%_%RANDOM%"
for %%E in (feature_baseline depth_weighted_loss attention adaptive_lr prior_only combined_best) do (
    python -X utf8 -u "%SCRIPT_DIR%run_experiment.py" --config "%SCRIPT_DIR%configs\%%E.yaml" --output-dir "%OUTPUT_BASE%" --device "%DEVICE%" --seed 42 --iterations "%UPDATES%"
    if errorlevel 1 exit /b 1
)
python -X utf8 "%SCRIPT_DIR%compare_results.py" --results-dir "%OUTPUT_BASE%"
exit /b %errorlevel%
