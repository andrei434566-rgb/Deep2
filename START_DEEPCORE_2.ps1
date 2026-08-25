$ErrorActionPreference = 'Stop'

function Show-LaunchError([string]$message) {
    Write-Host ''
    Write-Host 'DeepCore 2 did not start:' -ForegroundColor Red
    Write-Host $message -ForegroundColor Yellow
    Write-Host ''
    Read-Host 'Press Enter to close this window'
    exit 1
}

try {
    Set-Location $PSScriptRoot
    # Use the installed interpreter directly.  This avoids a Windows launcher
    # issue with virtual environments located inside a Cyrillic user profile.
    # Python 3.14 is the installed interpreter on this workstation.
    $python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        throw 'Python 3.11 or newer was not found. Install Python, then start DeepCore 2 again.'
    }

    Write-Host 'Checking DeepCore 2 modules...'
    & $python -m py_compile run.py app\ui\windows\main_window.py app\infrastructure\core_report_export.py
    if ($LASTEXITCODE -ne 0) { throw 'The application code check failed.' }

    & $python -c 'import PySide6, ultralytics, docx, reportlab, openpyxl' 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Installing DeepCore 2 components...'
        & $python -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the required components.' }
    }

    $pythonWindowless = Join-Path (Split-Path -Parent $python) 'pythonw.exe'
    if (Test-Path -LiteralPath $pythonWindowless) {
        Start-Process -FilePath $pythonWindowless -ArgumentList 'run.py' -WorkingDirectory $PSScriptRoot
    } else {
        Start-Process -FilePath $python -ArgumentList 'run.py' -WorkingDirectory $PSScriptRoot
    }
    Write-Host 'Check passed. DeepCore 2 is opening.'
}
catch {
    Show-LaunchError $_.Exception.Message
}
