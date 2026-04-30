# Run RAG chat (avoids pasting bash-style ">>" lines into PowerShell).
# Usage (from backend\rag):
#   .\run_query.ps1 "What measures has RBI taken to prevent digital payment fraud?"
# Optional: .\run_query.ps1 -Mode brd "..."
param(
    [ValidateSet("query", "brd")]
    [string]$Mode = "query",
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Question
)
Set-Location $PSScriptRoot
python chat.py $Mode "$Question"
