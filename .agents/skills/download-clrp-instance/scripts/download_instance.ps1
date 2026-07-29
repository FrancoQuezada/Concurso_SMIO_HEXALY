param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('\.txt$')]
    [string]$FileName,

    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$DestinationDirectory = "C:\Users\mmate\OneDrive\Desktop\Programazzione\Concurso_SMIO_HEXALY\data\official"
$DestinationPath = Join-Path $DestinationDirectory $FileName

try {
    if (-not (Test-Path -LiteralPath $DestinationDirectory)) {
        New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
    }

    if ((Test-Path -LiteralPath $DestinationPath) -and -not $Overwrite) {
        throw "El archivo ya existe: $DestinationPath. Usa -Overwrite solo si deseas reemplazarlo."
    }

    $TemporaryPath = "$DestinationPath.download"

    if (Test-Path -LiteralPath $TemporaryPath) {
        Remove-Item -LiteralPath $TemporaryPath -Force
    }

    Invoke-WebRequest `
        -Uri $Url `
        -OutFile $TemporaryPath `
        -UseBasicParsing `
        -MaximumRedirection 5

    if (-not (Test-Path -LiteralPath $TemporaryPath)) {
        throw "La descarga no produjo ningún archivo."
    }

    $DownloadedFile = Get-Item -LiteralPath $TemporaryPath

    if ($DownloadedFile.Length -eq 0) {
        throw "El archivo descargado está vacío."
    }

    $FirstLines = Get-Content -LiteralPath $TemporaryPath -TotalCount 10
    $Preview = $FirstLines -join "`n"

    if ($Preview -match '^\s*<(html|!doctype|Error|Code)') {
        throw "El contenido descargado parece una página HTML/XML o un mensaje de error."
    }

    Move-Item -LiteralPath $TemporaryPath -Destination $DestinationPath -Force

    $FinalFile = Get-Item -LiteralPath $DestinationPath

    Write-Host "Descarga completada."
    Write-Host "Ruta: $($FinalFile.FullName)"
    Write-Host "Tamaño: $($FinalFile.Length) bytes"
    Write-Host ""
    Write-Host "Primeras líneas:"
    Get-Content -LiteralPath $DestinationPath -TotalCount 10
}
catch {
    if (Test-Path -LiteralPath "$DestinationPath.download") {
        Remove-Item -LiteralPath "$DestinationPath.download" -Force
    }

    $Message = $_.Exception.Message

    if ($Message -match '403|Forbidden') {
        Write-Error "El servidor rechazó la descarga (403). La URL firmada probablemente expiró. Genera un enlace nuevo."
    }
    elseif ($Message -match '404|Not Found') {
        Write-Error "El archivo no fue encontrado en el servidor (404)."
    }
    else {
        Write-Error $Message
    }

    exit 1
}
