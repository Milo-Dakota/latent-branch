param([string]$Save = "saves/demo", [ValidateSet("mock", "http")][string]$Provider = "mock")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$gamePython = Get-Command python -ErrorAction SilentlyContinue
if ($gamePython) {
    $gameRuntime = $gamePython.Source
} else {
    $gameRuntime = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if (-not (Test-Path -LiteralPath $gameRuntime)) {
        throw 'Please install Python 3.10+ and add it to PATH.'
    }
}
$env:PYTHONUTF8 = '1'
& $gameRuntime -m narrative_game --save $Save --provider $Provider
exit $LASTEXITCODE
