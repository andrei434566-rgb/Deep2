Add-Type -AssemblyName System.Drawing

$userName = [string][char]0x042F
$sourceFolder = [string]::Concat('a', [char]0x0442, [char]0x0435, [char]0x0441, [char]0x0442)
$inputPath = 'C:\Users\' + $userName + '\Desktop\' + $sourceFolder + '\16.png'
$outputPath = Join-Path $PSScriptRoot 'core_16_facies_overlay.png'

$image = [System.Drawing.Image]::FromFile($inputPath)
$bitmap = New-Object System.Drawing.Bitmap $image.Width, $image.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.DrawImage($image, 0, 0, $image.Width, $image.Height)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

function Add-FaciesPolygon {
    param([int[][]]$Coordinates, [System.Drawing.Color]$Color)
    $points = [System.Drawing.Point[]]@($Coordinates | ForEach-Object {
        New-Object System.Drawing.Point($_[0], $_[1])
    })
    $brush = New-Object System.Drawing.SolidBrush $Color
    $pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(235, $Color.R, $Color.G, $Color.B)), 2
    $graphics.FillPolygon($brush, $points)
    $graphics.DrawPolygon($pen, $points)
    $brush.Dispose(); $pen.Dispose()
}

$sf = [System.Drawing.Color]::FromArgb(78, 255, 215, 0)
$msf = [System.Drawing.Color]::FromArgb(96, 0, 174, 239)
$mf = [System.Drawing.Color]::FromArgb(105, 46, 204, 113)

# Mf: massive mud packages
Add-FaciesPolygon @(@(30,36),@(88,36),@(88,171),@(30,171)) $mf
Add-FaciesPolygon @(@(478,600),@(538,600),@(538,733),@(478,733)) $mf

# Sf: pale massive sand packages in the second core
Add-FaciesPolygon @(@(177,42),@(234,42),@(234,130),@(177,130)) $sf
Add-FaciesPolygon @(@(177,267),@(234,267),@(234,315),@(177,315)) $sf
Add-FaciesPolygon @(@(177,481),@(234,481),@(234,651),@(177,651)) $sf

# Msf: rhythmically laminated mixed sand-mud packages
Add-FaciesPolygon @(@(30,171),@(88,171),@(88,733),@(30,733)) $msf
Add-FaciesPolygon @(@(177,130),@(234,130),@(234,267),@(177,267)) $msf
Add-FaciesPolygon @(@(177,315),@(234,315),@(234,481),@(177,481)) $msf
Add-FaciesPolygon @(@(177,651),@(234,651),@(234,733),@(177,733)) $msf
Add-FaciesPolygon @(@(282,14),@(337,14),@(337,733),@(282,733)) $msf
Add-FaciesPolygon @(@(382,37),@(438,37),@(438,733),@(382,733)) $msf
Add-FaciesPolygon @(@(478,35),@(538,35),@(538,600),@(478,600)) $msf

# Compact code legend in the gap between the first two boxes.
$legendX = 103; $legendY = 35
$background = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(225,255,255,255))
$border = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(190,70,70,70)), 1
$graphics.FillRectangle($background, $legendX, $legendY, 63, 78)
$graphics.DrawRectangle($border, $legendX, $legendY, 63, 78)
$font = New-Object System.Drawing.Font('Arial', 9, [System.Drawing.FontStyle]::Bold)
$textBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::Black)

function Add-LegendCode {
    param([int]$Y, [System.Drawing.Color]$Color, [string]$Text)
    $fill = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(180,$Color.R,$Color.G,$Color.B))
    $line = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(235,$Color.R,$Color.G,$Color.B)), 1
    $graphics.FillRectangle($fill, $legendX + 7, $Y, 15, 15)
    $graphics.DrawRectangle($line, $legendX + 7, $Y, 15, 15)
    $graphics.DrawString($Text, $font, $textBrush, $legendX + 28, $Y)
    $fill.Dispose(); $line.Dispose()
}

Add-LegendCode 45 ([System.Drawing.Color]::FromArgb(255,215,0)) 'Sf'
Add-LegendCode 68 ([System.Drawing.Color]::FromArgb(0,174,239)) 'Msf'
Add-LegendCode 91 ([System.Drawing.Color]::FromArgb(46,204,113)) 'Mf'

$bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$textBrush.Dispose(); $font.Dispose(); $border.Dispose(); $background.Dispose()
$graphics.Dispose(); $bitmap.Dispose(); $image.Dispose()
Write-Output $outputPath
