# Локальный запуск сканера — основной режим.
#
# Планировщик Windows дёргает его каждые 5 минут. Задержка получается в
# секунды после закрытия бара, тогда как GitHub Actions тормозит на 5-13
# минут и вдобавок выбрасывает часть запусков.
#
# После расчёта состояние уходит в репозиторий. Это не только резервная
# копия журнала: свежесть state.json служит признаком жизни для облачного
# сканера — пока файл свежий, облако молчит и дублей не шлёт.

Set-Location -LiteralPath $PSScriptRoot
$log = Join-Path $PSScriptRoot "run.log"
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

function Note($text) { Add-Content -LiteralPath $log -Value $text -Encoding utf8 }

# --- расчёт сигналов ---
$out = & python scan.py 2>&1 | Out-String
$scanCode = $LASTEXITCODE
Note "[$stamp]`n$out"

if ($scanCode -ne 0) {
    Note "сканер вернул код $scanCode — состояние не отправляю"
    exit $scanCode
}

# --- отправка состояния в репозиторий ---
#
# Здесь намеренно НЕ используется ErrorActionPreference = Stop и не
# перенаправляется stderr: git пишет туда обычные сообщения вроде
# "Applied autostash", а PowerShell 5.1 превращает такой вывод в
# терминирующую ошибку, хотя команда отработала успешно.
& git add signals.csv state.json | Out-Null
& git diff --staged --quiet
$nothingStaged = ($LASTEXITCODE -eq 0)

if (-not $nothingStaged) {
    & git commit -q -m "signals $(Get-Date -Format 'yyyy-MM-dd HH:mm') local" | Out-Null
    if ($LASTEXITCODE -ne 0) { Note "коммит не удался"; exit 1 }
}

# Пушим не только свежий коммит: если прошлый запуск не смог отправить,
# коммит лежит локально и облако продолжает считать компьютер выключенным.
$ahead = (& git rev-list --count "origin/main..HEAD")
if ($nothingStaged -and $ahead -eq "0") { exit 0 }

& git pull --rebase --autostash -q origin main | Out-Null
& git push -q | Out-Null
if ($LASTEXITCODE -eq 0) {
    Note "состояние отправлено в git"
} else {
    # Не критично: сигналы уже ушли в Telegram, а коммит лежит локально
    # и уедет со следующим запуском.
    Note "push не прошёл (код $LASTEXITCODE) — коммит остался локально"
}

# Лог не должен расти бесконечно
if ((Test-Path $log) -and (Get-Item $log).Length -gt 1MB) {
    $tail = Get-Content -LiteralPath $log -Tail 2000
    Set-Content -LiteralPath $log -Value $tail -Encoding utf8
}
exit 0
