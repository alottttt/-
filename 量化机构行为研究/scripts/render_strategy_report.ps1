param(
    [Parameter(Mandatory=$true)][string]$DataPath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$report = Get-Content -LiteralPath $DataPath -Raw -Encoding UTF8 | ConvertFrom-Json
$bitmap = New-Object System.Drawing.Bitmap 2700,3000
$canvas = [System.Drawing.Graphics]::FromImage($bitmap)
$canvas.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$canvas.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$canvas.Clear([System.Drawing.ColorTranslator]::FromHtml('#F3F5F8'))
$fontCache = @{}

function Draw-Text {
    param([string]$Text, [single]$X, [single]$Y, [single]$Size=30,
          [string]$Color='#24354B', [bool]$Bold=$false,
          [string]$Align='Near', [single]$Width=0, [single]$Height=100)
    $key = "$Size/$Bold"
    if (-not $fontCache.ContainsKey($key)) {
        $style = if ($Bold) {[System.Drawing.FontStyle]::Bold} else {[System.Drawing.FontStyle]::Regular}
        $fontCache[$key] = [System.Drawing.Font]::new('Microsoft YaHei', $Size, $style, [System.Drawing.GraphicsUnit]::Pixel)
    }
    $brush = [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml($Color))
    $format = [System.Drawing.StringFormat]::new()
    $format.Alignment = [System.Drawing.StringAlignment]::$Align
    if ($Width -gt 0) {
        $canvas.DrawString($Text, $fontCache[$key], $brush, [System.Drawing.RectangleF]::new($X,$Y,$Width,$Height), $format)
    } else {
        $canvas.DrawString($Text, $fontCache[$key], $brush, [System.Drawing.PointF]::new($X,$Y), $format)
    }
    $format.Dispose()
    $brush.Dispose()
}

function Draw-Box {
    param([single]$X,[single]$Y,[single]$Width,[single]$Height,[string]$Color)
    $brush = [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml($Color))
    $canvas.FillRectangle($brush,$X,$Y,$Width,$Height)
    $brush.Dispose()
}

try {
    Draw-Text '机构行为策略  |  效果与因子贡献' 120 85 66 '#24354B' $true
    Draw-Text "共同有效样本 $($report.start) — $($report.end)  ·  10Y 国债下一行情观测日方向预测" 120 190 33
    Draw-Text "$($report.pool_note)    数据截至 $($report.as_of)" 120 250 28 '#64748B'

    $headers = @('策略 · 窗口 / K','因子池','对 bp','错 bp','对减错 bp','胜率','正确 / 方向样本','零仓位日')
    $widths = @(467,295,246,246,320,246,369,271)
    $pen = [System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml('#DFE6EE'), 1)
    for ($r = 0; $r -le $report.rows.Count; $r++) {
        $y = 335 + $r * 72
        $x = 120
        if ($r -eq 0) { $cells=$headers; $bg='#24354B' }
        else {
            $cells=$report.rows[$r-1].cells
            $bg=if ($report.rows[$r-1].expanded) {'#E8F3EF'} else {'#FFFFFF'}
        }
        for ($c = 0; $c -lt $headers.Count; $c++) {
            Draw-Box $x $y $widths[$c] 72 $bg
            $canvas.DrawRectangle($pen, [single]$x, [single]$y, [single]$widths[$c], [single]72)
            $fg=if ($r -eq 0) {'#FFFFFF'} else {'#24354B'}
            if ($r -gt 0 -and $c -eq 4) { $fg=if ([double]$cells[$c] -ge 0) {'#14836B'} else {'#BF554B'} }
            Draw-Text $cells[$c] $x ($y+18) 29 $fg ($r -eq 0 -or $c -eq 4) 'Center' $widths[$c] 50
            $x += $widths[$c]
        }
    }
    $pen.Dispose()
    Draw-Text "每策略共同参评日数：$($report.days)；胜率 = 正确日数 / 方向样本。零仓位或目标为0的日期不进入分母。" 120 1015 26
    Draw-Text $report.limit_note 120 1060 27 '#BF554B' $false 'Near' 2460 75
    Draw-Text '哪些因子帮助了当前投票结果？' 120 1145 49 '#24354B' $true
    Draw-Text '按移除该因子一票、不补位的“对减错 bp”差值排序；每策略单独计算，因子贡献不可相加为总成绩。' 120 1220 28

    for ($p=0; $p -lt $report.panels.Count; $p++) {
        $panel=$report.panels[$p]
        $left=if ($p % 2 -eq 0) {120} else {1390}
        $top=if ($p -lt 2) {1300} else {2020}
        Draw-Box $left $top 1190 680 '#FFFFFF'
        Draw-Text $panel.title ($left+32) ($top+25) 36 '#24354B' $true
        Draw-Text '扩展池正向贡献前五  ·  次数仅表示共同样本内入选频率' ($left+32) ($top+85) 24 '#64748B'
        if ($panel.top.Count -eq 0) {
            Draw-Text '该口径下没有正向贡献因子' ($left+32) ($top+160) 30
            continue
        }
        $maxContribution=[double]$panel.top[0].'边际贡献bp'
        for ($j=0; $j -lt $panel.top.Count; $j++) {
            $d=$panel.top[$j]
            $y=$top+145+$j*95
            $label=$d.'因子名' -replace '^识别_',''
            $parts=$label.Split('_')
            if ($parts.Count -ge 3) {
                $line1=$parts[0..($parts.Count-2)] -join ' / '
                $line2=$parts[-1]
                Draw-Text $line1 ($left+32) $y 27
                Draw-Text $line2 ($left+32) ($y+35) 27
            } else { Draw-Text $label ($left+32) ($y+16) 27 }
            $value=[double]$d.'边际贡献bp'
            Draw-Text ('+{0:F2} bp' -f $value) ($left+795) ($y-8) 29 '#14836B' $true 'Far' 335 42
            Draw-Box ($left+795) ($y+34) (335*$value/$maxContribution) 11 '#14836B'
            Draw-Text ('入选 {0:N0} 次' -f $d.'入选次数') ($left+795) ($y+49) 23 '#64748B' $false 'Far' 335 35
        }
        if ($panel.worst.Count -gt 0 -and [double]$panel.worst[0].'边际贡献bp' -lt 0) {
            $worst=$panel.worst[0]
            Draw-Text ('最大负向：'+($worst.'因子名' -replace '^识别_','')) ($left+32) ($top+615) 22 '#BF554B'
            Draw-Text ('{0:F2} bp' -f $worst.'边际贡献bp') ($left+795) ($top+646) 24 '#BF554B' $false 'Far' 335 34
        }
    }
    Draw-Text '对bp / 错bp：判断正确 / 错误日的收益率变动绝对值之和。多债(+1)对应收益率下降，空债(-1)对应上升。' 120 2760 26
    Draw-Text '因子贡献：逐日 −[原仓位 − sign(总票数−该票)] × 目标变动bp，再对共同样本求和；仅解释当前固定入选集合。' 120 2810 26
    Draw-Text '这是方向预测统计，尚未包含可执行入场价格和交易成本。识别族省略“识别_”前缀。报告由逐日记录重算并核对汇总。' 120 2860 25 '#64748B'
    Draw-Text "运行批次：$($report.run_id)" 120 2920 23 '#64748B'
    $encoder=[System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object {$_.MimeType -eq 'image/jpeg'}
    $parameters=[System.Drawing.Imaging.EncoderParameters]::new(1)
    $parameters.Param[0]=[System.Drawing.Imaging.EncoderParameter]::new([System.Drawing.Imaging.Encoder]::Quality,[long]95)
    try { $bitmap.Save($OutputPath,$encoder,$parameters) }
    finally { $parameters.Dispose() }
} finally {
    foreach ($font in $fontCache.Values) { $font.Dispose() }
    $canvas.Dispose()
    $bitmap.Dispose()
}
