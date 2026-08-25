Add-Type -AssemblyName System.Drawing

$userName = [string][char]0x042F
$sourceFolder = [string]::Concat('a', [char]0x0442, [char]0x0435, [char]0x0441, [char]0x0442)
$sourceFile = [string]::Concat([char]0x0423, [char]0x0420, '-1.jpg')
$inputPath = 'C:\Users\' + $userName + '\Desktop\' + $sourceFolder + '\' + $sourceFile
$outputPath = Join-Path $PSScriptRoot 'core_facies_overlay.png'

$image = [System.Drawing.Image]::FromFile($inputPath)
$bitmap = New-Object System.Drawing.Bitmap $image.Width, $image.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.DrawImage($image, 0, 0, $image.Width, $image.Height)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

function Add-FaciesPolygon {
    param(
        [int[][]]$Coordinates,
        [System.Drawing.Color]$Color
    )
    $points = [System.Drawing.Point[]]@($Coordinates | ForEach-Object {
        New-Object System.Drawing.Point($_[0], $_[1])
    })
    $brush = New-Object System.Drawing.SolidBrush $Color
    $pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(235, $Color.R, $Color.G, $Color.B)), 2
    $graphics.FillPolygon($brush, $points)
    $graphics.DrawPolygon($pen, $points)
    $brush.Dispose()
    $pen.Dispose()
}

$yellow = [System.Drawing.Color]::FromArgb(82, 255, 215, 0)
$cyan = [System.Drawing.Color]::FromArgb(105, 0, 174, 239)
$green = [System.Drawing.Color]::FromArgb(105, 46, 204, 113)

# Fchm - meandering river channel
Add-FaciesPolygon @(@(268,160),@(341,160),@(341,310),@(268,310)) $yellow
Add-FaciesPolygon @(@(268,351),@(341,351),@(341,1028),@(268,1028)) $yellow
Add-FaciesPolygon @(@(404,77),@(480,77),@(480,726),@(404,726)) $yellow
Add-FaciesPolygon @(@(542,78),@(618,78),@(618,1028),@(542,1028)) $yellow

# Lms - lacustrine mud
Add-FaciesPolygon @(@(268,77),@(341,77),@(341,156),@(329,158),@(316,157),@(307,160),@(268,159)) $cyan
Add-FaciesPolygon @(@(268,310),@(341,310),@(341,351),@(268,351)) $cyan
Add-FaciesPolygon @(@(404,726),@(480,726),@(480,948),@(470,948),@(470,955),@(404,955)) $cyan

# Fp - flood plain
Add-FaciesPolygon @(@(404,947),@(470,947),@(480,955),@(480,1038),@(404,1038)) $green

# Legend in clear white area.
$legendX = 630; $legendY = 150; $legendWidth = 195; $legendHeight = 120
$background = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(235,255,255,255))
$border = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(180,60,60,60)), 1
$graphics.FillRectangle($background, $legendX, $legendY, $legendWidth, $legendHeight)
$graphics.DrawRectangle($border, $legendX, $legendY, $legendWidth, $legendHeight)
$font = New-Object System.Drawing.Font('Arial', 10, [System.Drawing.FontStyle]::Regular)
$titleFont = New-Object System.Drawing.Font('Arial', 11, [System.Drawing.FontStyle]::Bold)
$textBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::Black)
$graphics.DrawString('Facies', $titleFont, $textBrush, $legendX + 10, $legendY + 8)

function Add-LegendItem {
    param([int]$Y, [System.Drawing.Color]$Color, [string]$Text)
    $fill = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(170,$Color.R,$Color.G,$Color.B))
    $line = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(235,$Color.R,$Color.G,$Color.B)), 1
    $graphics.FillRectangle($fill, $legendX + 10, $Y, 18, 18)
    $graphics.DrawRectangle($line, $legendX + 10, $Y, 18, 18)
    $graphics.DrawString($Text, $font, $textBrush, $legendX + 36, $Y + 1)
    $fill.Dispose(); $line.Dispose()
}

Add-LegendItem ($legendY + 35) ([System.Drawing.Color]::FromArgb(255,215,0)) 'Fchm - meandering channel'
Add-LegendItem ($legendY + 59) ([System.Drawing.Color]::FromArgb(0,174,239)) 'Lms - lacustrine mud'
Add-LegendItem ($legendY + 83) ([System.Drawing.Color]::FromArgb(46,204,113)) 'Fp - flood plain'

$bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)

$textBrush.Dispose(); $titleFont.Dispose(); $font.Dispose(); $border.Dispose(); $background.Dispose()
$graphics.Dispose(); $bitmap.Dispose(); $image.Dispose()
Write-Output $outputPath
