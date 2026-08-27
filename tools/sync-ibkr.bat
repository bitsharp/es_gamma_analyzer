@echo off
rem ---------------------------------------------------------------------------
rem Lanciato dalle attivita' pianificate di Polaris. Gli argomenti passano dritti
rem allo script Python, quindi lo stesso file serve due usi:
rem
rem   sync-ibkr.bat              giro completo: gateway (ordini, capitale, P&L
rem                              del giorno) piu' il giro Flex. Una volta la
rem                              mattina, quando il gateway e' appena stato
rem                              autenticato a mano.
rem   sync-ibkr.bat --flex-only  solo posizioni, lato server. Ogni mezz'ora:
rem                              non tocca il gateway, quindi non serve che sia
rem                              acceso ne' autenticato.
rem
rem Esiste per tenere un log: uno script pianificato gira in silenzio, e senza
rem log il giorno in cui qualcosa non arriva non si capirebbe perche'.
rem ---------------------------------------------------------------------------

setlocal
set "ROOT=%~dp0.."
set "LOG=%ROOT%\tools\sync-ibkr.log"
set "PY=C:\Users\lucag\AppData\Local\Programs\Python\Python312\python.exe"

if not exist "%PY%" set "PY=python"

set "MODO=gateway+flex"
if "%~1"=="--flex-only" set "MODO=solo flex"

echo. >> "%LOG%"
echo ===== %DATE% %TIME% [%MODO%] ===== >> "%LOG%"
"%PY%" "%ROOT%\tools\ibkr_gateway_sync.py" %* >> "%LOG%" 2>&1
echo (uscita: %ERRORLEVEL%) >> "%LOG%"

rem Il log non deve crescere all'infinito: sopra 200 KB si tiene solo la coda.
for %%F in ("%LOG%") do if %%~zF GTR 200000 (
    more +200 "%LOG%" > "%LOG%.tmp" && move /y "%LOG%.tmp" "%LOG%" >nul
)
endlocal
