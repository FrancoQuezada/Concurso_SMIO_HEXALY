---
name: download-clrp-instance
description: Descarga una instancia oficial CLRP desde una URL proporcionada por el usuario y la guarda en la carpeta data/official del repositorio Concurso_SMIO_HEXALY. Úsala cuando el usuario entregue un enlace de descarga temporal de una instancia .txt y solicite guardarla localmente.
---

# Descargar instancia oficial CLRP

Descarga una instancia `.txt` desde la URL entregada por el usuario y guárdala en:

`C:\Users\mmate\OneDrive\Desktop\Programazzione\Concurso_SMIO_HEXALY\data\official`

## Procedimiento

1. Obtén de la solicitud del usuario:
   - La URL completa.
   - El nombre del archivo de destino, preferentemente deducido de la ruta de la URL.
2. No reutilices URLs firmadas antiguas. Los parámetros `X-Amz-Date` y `X-Amz-Expires` indican que el enlace puede expirar rápidamente.
3. Verifica que el nombre termine en `.txt`.
4. Ejecuta el script `scripts/download_instance.ps1` pasando:
   - `-Url` con la URL completa.
   - `-FileName` con el nombre final.
5. Después de descargar:
   - Comprueba que el archivo exista.
   - Comprueba que no esté vacío.
   - Muestra la ruta final y el tamaño del archivo.
   - Lee solamente las primeras 10 líneas para verificar que parece una instancia CLRP.
6. No sobrescribas un archivo existente silenciosamente. El script debe detenerse, salvo que el usuario haya solicitado explícitamente reemplazarlo y se use `-Overwrite`.
7. No modifiques ningún otro archivo del repositorio.

## Ejemplo de uso

Cuando el usuario escriba:

`Usa $download-clrp-instance para descargar este enlace: <URL>`

deduce un nombre como `clrp-medium-10.txt` y ejecuta:

```powershell
powershell -ExecutionPolicy Bypass -File ".agents\skills\download-clrp-instance\scripts\download_instance.ps1" `
  -Url "<URL_COMPLETA>" `
  -FileName "clrp-medium-10.txt"
```

Si el usuario autoriza reemplazar un archivo existente:

```powershell
powershell -ExecutionPolicy Bypass -File ".agents\skills\download-clrp-instance\scripts\download_instance.ps1" `
  -Url "<URL_COMPLETA>" `
  -FileName "clrp-medium-10.txt" `
  -Overwrite
```

## Manejo de errores

- Si el servidor responde `403`, informa que probablemente la URL firmada expiró y solicita un enlace nuevo.
- Si el servidor responde `404`, informa que el recurso no fue encontrado.
- Si falla la conexión, no crees un archivo vacío.
- Si el contenido descargado parece HTML, XML o un mensaje de error, elimina el archivo incompleto y explica el problema.
