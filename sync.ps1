param(
  [string]$Message = "",
  [switch]$PullOnly,
  [switch]$Verbose
)

if ($Message -eq "" -and -not $PullOnly) {
  $Message = "Update " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

if (-not $env:HF_TOKEN -or $env:HF_TOKEN.Trim().Length -lt 10) {
  Write-Host "Error: HF_TOKEN not set"
  Write-Host "Set with: [Environment]::SetEnvironmentVariable('HF_TOKEN','YOUR_TOKEN','User')"
  exit 1
}

$repoRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Error: Not a git repository"
  exit 1
}
Set-Location $repoRoot

$hfRemote = git remote get-url huggingface 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Error: HuggingFace remote not configured"
  exit 1
}

if ($hfRemote -match 'huggingface\.co/spaces/([^/]+)/(.+?)(?:\.git)?$') {
  $hfUser = $Matches[1]
  $hfSpace = $Matches[2]
} else {
  Write-Host "Error: Cannot parse HF remote"
  exit 1
}

# Load config
$config = @{ max_file_size_mb = 10; ask_before_upload = $true }
if (Test-Path ".syncconfig") {
  try {
    $configData = Get-Content ".syncconfig" -Raw | ConvertFrom-Json
    $config.max_file_size_mb = $configData.max_file_size_mb
    $config.ask_before_upload = $configData.ask_before_upload
  } catch {}
}

Write-Host "`n=== 3-Way Date-Priority Sync ===`n"

function Get-RemoteCommit($remote) {
  $commit = git rev-parse "$remote/main" 2>$null
  if ($LASTEXITCODE -eq 0) { return $commit }
  return $null
}

function Get-RemoteDate($remote) {
  $date = git log $remote/main -1 --format=%ci 2>$null
  if ($LASTEXITCODE -eq 0) { return [datetime]$date }
  return [datetime]::MinValue
}

function Get-FileSize($path) {
  if (Test-Path $path) {
    return (Get-Item $path).Length / 1MB
  }
  return 0
}

function Ask-Upload($fileName, $sizeMB) {
  Write-Host "`n⚠️  Large file: $fileName ($([math]::Round($sizeMB, 2)) MB)"
  Write-Host "1. Yes - Upload this file"
  Write-Host "2. No - Skip this file"
  Write-Host "3. All - Upload all large files"
  Write-Host "4. Skip - Skip all large files"
  
  $choice = Read-Host "Choice [1-4] (default: 2)"
  if ($choice -eq "") { $choice = "2" }
  
  return $choice
}

# Stage changes
git add -A

# Check for large files
$largeFiles = @()
$allFiles = git diff --cached --name-only --diff-filter=ACMR
$skipAll = $false
$uploadAll = $false

if ($config.ask_before_upload) {
  foreach ($file in $allFiles) {
    if (Test-Path $file) {
      $sizeMB = Get-FileSize $file
      if ($sizeMB -gt $config.max_file_size_mb) {
        $largeFiles += @{ Path = $file; Size = $sizeMB }
      }
    }
  }
  
  if ($largeFiles.Count -gt 0) {
    Write-Host "`n📊 Found $($largeFiles.Count) large file(s) (>$($config.max_file_size_mb) MB)"
    
    $filesToSkip = @()
    
    foreach ($fileInfo in $largeFiles) {
      if ($uploadAll) { continue }
      if ($skipAll) {
        $filesToSkip += $fileInfo.Path
        continue
      }
      
      $choice = Ask-Upload $fileInfo.Path $fileInfo.Size
      
      switch ($choice) {
        "1" { continue }
        "2" { $filesToSkip += $fileInfo.Path }
        "3" { $uploadAll = $true }
        "4" { 
          $skipAll = $true
          $filesToSkip += $fileInfo.Path
        }
      }
    }
    
    if ($filesToSkip.Count -gt 0) {
      Write-Host "`n⏭️  Skipping $($filesToSkip.Count) file(s) this sync:"
      foreach ($f in $filesToSkip) {
        git reset HEAD $f 2>$null | Out-Null
        Write-Host "  - $f"
      }
    }
  }
}

# Fetch
Write-Host "`nFetching from remotes..."
git fetch github 2>$null
$ghExists = $LASTEXITCODE -eq 0

git fetch huggingface 2>$null  
$hfExists = $LASTEXITCODE -eq 0

if (-not $ghExists -and -not $hfExists) {
  $isFirstPush = $true
} else {
  $isFirstPush = $false
}

# Merge
if (-not $isFirstPush) {
  $localCommit = git rev-parse HEAD 2>$null
  $ghCommit = Get-RemoteCommit "github"
  $hfCommit = Get-RemoteCommit "huggingface"
  
  $localDate = git log HEAD -1 --format=%ci 2>$null
  $localDateTime = if ($localDate) { [datetime]$localDate } else { [datetime]::MinValue }
  
  $ghDate = Get-RemoteDate "github"
  $hfDate = Get-RemoteDate "huggingface"
  
  Write-Host "Last Update Dates:"
  Write-Host "  Local:        $($localDateTime.ToString('yyyy-MM-dd HH:mm:ss'))"
  if ($ghExists) { Write-Host "  GitHub:       $($ghDate.ToString('yyyy-MM-dd HH:mm:ss'))" }
  if ($hfExists) { Write-Host "  HuggingFace:  $($hfDate.ToString('yyyy-MM-dd HH:mm:ss'))" }
  Write-Host ""
  
  $remotesToMerge = @()
  
  if ($ghExists -and $ghCommit -and $localCommit -ne $ghCommit) {
    $remotesToMerge += @{Name="github"; Date=$ghDate; Commit=$ghCommit}
  }
  
  if ($hfExists -and $hfCommit -and $localCommit -ne $hfCommit) {
    $remotesToMerge += @{Name="huggingface"; Date=$hfDate; Commit=$hfCommit}
  }
  
  $remotesToMerge = $remotesToMerge | Sort-Object -Property Date -Descending
  
  foreach ($remote in $remotesToMerge) {
    $remoteName = $remote.Name
    $remoteDate = $remote.Date.ToString('yyyy-MM-dd HH:mm:ss')
    
    Write-Host "Merging from $remoteName (updated: $remoteDate)..."
    
    git diff HEAD --quiet
    if ($LASTEXITCODE -ne 0) {
      git commit -m "Local changes before $remoteName merge" 2>$null | Out-Null
    }
    
    git merge "$remoteName/main" --no-edit 2>&1 | Out-Null
    
    if ($LASTEXITCODE -ne 0) {
      Write-Host "[!] Conflict. Keeping local version..."
      git merge --abort 2>$null
    } else {
      Write-Host "[OK] Merged from $remoteName"
    }
    
    $localCommit = git rev-parse HEAD 2>$null
  }
  
  $localCommit = git rev-parse HEAD 2>$null
  if ($ghCommit -and $hfCommit -and $localCommit -eq $ghCommit -and $localCommit -eq $hfCommit) {
    Write-Host "[OK] All 3 locations in sync"
  } elseif ($remotesToMerge.Count -eq 0) {
    Write-Host "[OK] Already in sync"
  }
}

if ($PullOnly) {
  Write-Host "`n[OK] Pull complete`n"
  exit 0
}

# Commit
git diff HEAD --quiet
if ($LASTEXITCODE -ne 0) {
  git commit -m $Message
  Write-Host "[OK] Committed: $Message"
} else {
  $localCommit = git rev-parse HEAD 2>$null
  $needsPush = $false
  
  if ($ghExists) {
    $ghCommit = Get-RemoteCommit "github"
    if ($localCommit -ne $ghCommit) { $needsPush = $true }
  } else {
    $needsPush = $true
  }
  
  if ($hfExists) {
    $hfCommit = Get-RemoteCommit "huggingface"
    if ($localCommit -ne $hfCommit) { $needsPush = $true }
  } else {
    $needsPush = $true
  }
  
  if (-not $needsPush) {
    Write-Host "`n[OK] Already up-to-date`n"
    exit 0
  }
}

# Push
Write-Host "`nPushing to remotes..."

git push github main 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
  Write-Host "[OK] GitHub"
} else {
  git push github main --force 2>&1 | Out-Null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] GitHub (forced)"
  } else {
    Write-Host "[ERROR] GitHub failed"
  }
}

$hfUrl = "https://${hfUser}:$($env:HF_TOKEN)@huggingface.co/spaces/${hfUser}/${hfSpace}"

if ($Verbose) {
  $pushOutput = git push $hfUrl main --force 2>&1
  Write-Host "Git output: $pushOutput"
} else {
  $pushOutput = git push $hfUrl main --force 2>&1
}

if ($LASTEXITCODE -eq 0) {
  Write-Host "[OK] HuggingFace"
} else {
  Write-Host "[ERROR] HuggingFace FAILED!"
  Write-Host "`nError: $pushOutput"
  Write-Host "`nCheck:"
  Write-Host "1. Space exists? huggingface.co/spaces/${hfUser}/${hfSpace}"
  Write-Host "2. Token valid?"
  Write-Host "3. Try: .\sync.ps1 -Verbose"
  exit 1
}

Write-Host "`n[OK] Sync complete!`n"
