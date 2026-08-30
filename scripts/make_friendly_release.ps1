param(
    [string]$Source = "$PSScriptRoot\..\dist\Kern_Analyzer",
    [string]$Output = "$PSScriptRoot\..\dist"
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$sourceRoot = (Resolve-Path -LiteralPath $Source).Path
$outputRoot = (Resolve-Path -LiteralPath $Output).Path
$archiveNames = @(
    'Kern-Analyzer-v1.5-Windows-part1.zip',
    'Kern-Analyzer-v1.5-Windows-part2.zip'
)

$files = Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | Where-Object {
    $relative = $_.FullName.Substring($sourceRoot.Length).TrimStart('\')
    -not $relative.StartsWith('_payload_chunks\', [System.StringComparison]::OrdinalIgnoreCase) -and
    # These DLLs can be collected from the build machine PATH. They conflict
    # with the Qt bundle on Windows 10 LTSC and must not be shipped.
    $relative -notin @('_internal\icuuc.dll', '_internal\icudt78.dll')
}

if (-not ($files.Name -contains 'Kern_Analyzer.exe')) {
    throw 'Kern_Analyzer.exe is missing from the source folder.'
}

# Largest-first packing keeps both release assets comfortably below GitHub's
# 2 GB per-asset limit while retaining the original folder structure.
$bins = @(@(), @())
$sizes = @(0L, 0L)
$mainExe = $files | Where-Object { $_.FullName -eq (Join-Path $sourceRoot 'Kern_Analyzer.exe') }
$bins[0] += $mainExe
$sizes[0] += $mainExe.Length

$files | Where-Object { $_.FullName -ne $mainExe.FullName } |
    Sort-Object Length -Descending |
    ForEach-Object {
        $target = if ($sizes[0] -le $sizes[1]) { 0 } else { 1 }
        $bins[$target] += $_
        $sizes[$target] += $_.Length
    }

function Add-TextEntry([System.IO.Compression.ZipArchive]$archive, [string]$path, [string]$contents) {
    $entry = $archive.CreateEntry($path, [System.IO.Compression.CompressionLevel]::Optimal)
    $writer = [System.IO.StreamWriter]::new($entry.Open(), [System.Text.UTF8Encoding]::new($false))
    try { $writer.Write($contents) } finally { $writer.Dispose() }
}

for ($index = 0; $index -lt 2; $index++) {
    $archivePath = Join-Path $outputRoot $archiveNames[$index]
    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath -Force
    }

    $archive = [System.IO.Compression.ZipFile]::Open($archivePath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($file in $bins[$index]) {
            $relative = $file.FullName.Substring($sourceRoot.Length).TrimStart('\').Replace('\', '/')
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive,
                $file.FullName,
                "Kern_Analyzer/$relative",
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }

        if ($index -eq 0) {
            Add-TextEntry $archive 'Kern_Analyzer/README.txt' @"
Kern Analyzer v1.5 — full Windows installation

1. Download both files: part1 and part2.
2. Extract BOTH archives into the same place. They will form one folder named Kern_Analyzer.
3. Open Kern_Analyzer and run Kern_Analyzer.exe. Without parameters it opens the graphical app.
4. Optional BAT shortcuts run the core-tape builder, Excel facies masks, audit, and synthetic DEMO.

This is the full GPU-enabled release. Keep the complete folder: _internal and tools are required for model training and offline Russian OCR.
"@
        }
    }
    finally {
        $archive.Dispose()
    }
}

Get-ChildItem -LiteralPath $outputRoot -File -Filter 'Kern-Analyzer-v1.5-Windows-part*.zip' |
    Sort-Object Name |
    Select-Object Name, Length, @{Name='SHA256'; Expression={(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}}
