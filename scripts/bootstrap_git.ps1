param(
  [string]$RemoteUrl = "",
  [string]$DefaultBranch = "main"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
  git init
}

git branch -M $DefaultBranch

if ($RemoteUrl) {
  $existingRemotes = @(git remote)
  if ($existingRemotes -contains "origin") {
    git remote set-url origin $RemoteUrl
  }
  else {
    git remote add origin $RemoteUrl
  }
}

Write-Host "Git bootstrap completed." -ForegroundColor Green
git remote -v
git status -sb
