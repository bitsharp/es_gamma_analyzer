@echo off
rem ---------------------------------------------------------------------------
rem Lanciato dall'attivita' pianificata "Polaris - Sync IBKR" ogni sera.
rem
rem Esiste per una ragione sola: tenere un log. Uno script pianificato che gira
rem con pythonw non mostra niente, e la sessione del gateway cade da sola dopo
rem qualche ora — senza log, il giorno in cui gli ordini non arrivano non si
rem capirebbe perche'.
rem ---------------------------------------------------------------------------

setlocal
set "ROOT=%~dp0.."
set "LOG=%ROOT%\tools\sync-ibkr.log"
set "PY=C:\Users\lucag\AppData\Local\Programs\Python\Python312\python.exe"

if not exist "%PY%" set "PY=python"

echo. >> "%LOG%"
echo ===== %DATE% %TIME% ===== >> "%LOG%"
"%PY%" "%ROOT%\tools\ibkr_gateway_sync.py" >> "%LOG%" 2>&1
echo (uscita: %ERRORLEVEL%) >> "%LOG%"

rem Il log non deve crescere all'infinito: sopra 200 KB si tiene solo la coda.
for %%F in ("%LOG%") do if %%~zF GTR 200000 (
    more +200 "%LOG%" > "%LOG%.tmp" && move /y "%LOG%.tmp" "%LOG%" >nul
)
endlocal
