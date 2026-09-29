@echo off
REM Run all IFWI improvement experiments on Windows
REM Usage: run_all_experiments.bat [device]

SET DEVICE=%1
IF "%DEVICE%"=="" SET DEVICE=cuda:0

SET SCRIPT_DIR=%~dp0
SET OUTPUT_BASE=%SCRIPT_DIR%..\results

echo =========================================
echo Running All IFWI Improvement Experiments
echo =========================================
echo Device: %DEVICE%
echo Output directory: %OUTPUT_BASE%
echo.

REM List of experiments
SET EXPERIMENTS=baseline depth_weighted_loss attention adaptive_lr combined_best

FOR %%E IN (%EXPERIMENTS%) DO (
    echo.
    echo =========================================
    echo Starting Experiment: %%E
    echo =========================================

    SET CONFIG=%SCRIPT_DIR%configs\%%E.yaml

    IF NOT EXIST "%CONFIG%" (
        echo ERROR: Config file not found: %CONFIG%
        GOTO :NEXT
    )

    python "%SCRIPT_DIR%run_experiment.py" --config "%CONFIG%" --output-dir "%OUTPUT_BASE%" --device %DEVICE% --seed 42

    IF %ERRORLEVEL% EQU 0 (
        echo Success: Experiment '%%E' completed
    ) ELSE (
        echo Failed: Experiment '%%E' failed
    )

    :NEXT
)

echo.
echo =========================================
echo EXPERIMENTS COMPLETE
echo =========================================
echo Results saved to: %OUTPUT_BASE%
echo.
echo Generating comparison report...
python "%SCRIPT_DIR%compare_results.py" --results-dir "%OUTPUT_BASE%"

pause
