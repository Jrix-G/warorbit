$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Target = "analysis/v15_action_dataset.npz"
$TargetWindows = $Target -replace "/", "\"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupBranch = "backup/before-large-file-cleanup-$Timestamp"
$BackupPath = Join-Path $RepoRoot ".tmp\large_file_backup\$TargetWindows"

if ((git rev-parse --is-inside-work-tree).Trim() -ne "true") {
  throw "Not inside a git worktree: $RepoRoot"
}

$dirty = @(git status --porcelain)
if ($dirty.Count -gt 0) {
  Write-Host "Working tree is not clean. Commit current changes before rewriting history:" -ForegroundColor Yellow
  Write-Host "  git add .gitignore scripts/fix_github_large_file.ps1"
  Write-Host "  git commit -m `"Ignore generated analysis artifacts`""
  throw "Aborting because git filter-branch requires a clean working tree."
}

$upstream = ""
try {
  $upstream = (git rev-parse --abbrev-ref --symbolic-full-name "@{u}").Trim()
} catch {
  $upstream = "origin/main"
}
if (-not $upstream) {
  $upstream = "origin/main"
}

Write-Host "Creating backup branch: $BackupBranch"
git branch $BackupBranch

if (Test-Path $TargetWindows) {
  $backupDir = Split-Path -Parent $BackupPath
  New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
  Copy-Item -LiteralPath $TargetWindows -Destination $BackupPath -Force
  Write-Host "Backed up local file to: $BackupPath"
}

Write-Host "Removing $Target from local commits ahead of $upstream"
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch $Target" --prune-empty -- "$upstream..HEAD"

$originalRefs = @(git for-each-ref --format="%(refname)" refs/original/)
foreach ($ref in $originalRefs) {
  if ($ref) {
    git update-ref -d $ref
  }
}

if ((Test-Path $BackupPath) -and -not (Test-Path $TargetWindows)) {
  $targetDir = Split-Path -Parent $TargetWindows
  New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
  Copy-Item -LiteralPath $BackupPath -Destination $TargetWindows -Force
  Write-Host "Restored local untracked file: $TargetWindows"
}

Write-Host ""
Write-Host "Done. Review with:"
Write-Host "  git status --short"
Write-Host "  git rev-list --objects HEAD ^$upstream | Select-String -Pattern '$Target'"
Write-Host ""
Write-Host "If the verification command prints nothing, push:"
Write-Host "  git push"
