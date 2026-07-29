---
name: beat-clrp-benchmark
description: Busca una solución factible para una instancia del concurso SMIO CLRP cuyo costo sea estrictamente menor que un valor objetivo indicado por el usuario. Actívala con una entrada compacta como f(6000, medium-10), donde 6000 es el récord o mejor solución rival que se debe superar y medium-10 identifica la instancia.
---

# Superar un benchmark del concurso SMIO CLRP

## Contrato de entrada

La solicitud del usuario tendrá esta forma:

```text
f(mejor_solucion, instancia)
```

Ejemplos válidos:

```text
f(6000, medium-10)
f(6000, clrp-medium-10)
f(6000, clrp-medium-10.txt)
f(121909.88, small-asym-depot-binding)
```

Interpreta los argumentos así:

- `mejor_solucion`: costo objetivo publicado por otros concursantes. Es un número que se debe superar; no es una ruta ni un archivo de solución.
- `instancia`: nombre, alias o nombre de archivo de una instancia existente en el repositorio.

El objetivo es encontrar una solución **factible** con:

```text
costo_encontrado < mejor_solucion
```

Una igualdad no cuenta como mejora.

No solicites al usuario rutas, algoritmo, semilla, tiempo límite ni nombre de salida. Resuelve esos elementos automáticamente desde el repositorio.

## 1. Resolver la instancia automáticamente

Busca primero en:

```text
data/official
```

Si no aparece allí, busca recursivamente en:

```text
data
```

Normaliza el identificador recibido:

- Ignora mayúsculas y minúsculas.
- Considera equivalentes espacios, `_` y `-`.
- Acepta el nombre con o sin `.txt`.
- Acepta alias como `medium-10` para archivos como `clrp-medium-10.txt`.

Prioridad de coincidencia:

1. Nombre exacto.
2. Nombre exacto normalizado.
3. Coincidencia única que contenga el alias.
4. Coincidencia por número y categoría de instancia.

No adivines cuando existan varias coincidencias igualmente plausibles. En ese caso, termina indicando los candidatos encontrados.

## 2. Comprender el repositorio antes de ejecutar

Localiza y reutiliza lo que ya existe:

- Parser de instancias.
- Parser o escritor de soluciones.
- Evaluador del costo objetivo.
- Validador o checker de factibilidad.
- CLI del proyecto.
- Algoritmos constructivos, búsqueda local, ALNS, ILS, VNS, clustering, matheurísticas u otros métodos disponibles.
- Soluciones previas de la misma instancia.
- Configuraciones de semillas y límites de tiempo.

No inventes un formato alternativo si el repositorio ya define uno.

Configura el entorno del repositorio cuando corresponda, por ejemplo:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

## 3. Revisar soluciones locales existentes

Antes de ejecutar nuevas búsquedas:

1. Encuentra todas las soluciones locales asociadas a la instancia.
2. Valida cada candidata con el checker del repositorio.
3. Recalcula su costo.
4. Descarta archivos infactibles o corruptos.
5. Conserva la mejor solución local factible como incumbente.

Si una solución local ya cumple:

```text
costo_local < mejor_solucion
```

no gastes tiempo innecesario. Valídala nuevamente, guárdala o cópiala con un nombre inequívoco y reporta que el objetivo ya estaba superado.

## 4. Estrategia automática de optimización

Si todavía no se supera el objetivo, trabaja de forma iterativa.

### Fase A: explotar el solver actual

- Ejecuta primero el algoritmo más fuerte que ya exista en el repositorio.
- Usa múltiples semillas.
- Conserva siempre la mejor solución factible.
- Evita repetir configuraciones idénticas.
- Compara cada resultado con el incumbente y con el objetivo rival.
- Detén las ejecuciones restantes cuando ya exista una solución validada que supere el objetivo.

Configuración predeterminada cuando el repositorio no defina otra:

```text
Semillas iniciales: 1 a 10
Tiempo por ejecución: 300 segundos
Presupuesto total inicial: 1800 segundos
```

Puede redistribuir el presupuesto hacia las configuraciones que muestren mejores resultados.

### Fase B: diagnosticar la instancia

Analiza los factores que dominan el costo:

- Apertura y capacidad de depósitos.
- Límite de vehículos por depósito.
- Costo fijo por ruta.
- Utilización de vehículos.
- Asignación cliente-depósito.
- Número de rutas.
- Secuencia dentro de cada ruta.
- Matriz simétrica o asimétrica.
- Formato `COORDS` o `FULL_MATRIX`.
- Clientes frontera entre depósitos.
- Rutas caras, poco cargadas o fácilmente fusionables.

No uses supuestos euclidianos para una instancia `FULL_MATRIX`.

### Fase C: mejorar el método cuando sea necesario

Si las ejecuciones actuales no alcanzan el objetivo, realiza cambios pequeños y dirigidos en el solver. Prioriza:

- Mejor inicialización o clustering.
- Multi-start con diversificación real.
- Relocate intra e interruta.
- Swap.
- Or-opt.
- 2-opt para casos compatibles.
- Operadores válidos para matrices asimétricas.
- 2-opt* y cross-exchange.
- Merge o split de rutas.
- Reasignación entre depósitos.
- Apertura o cierre de depósitos.
- Destroy-and-repair.
- Selección adaptativa de operadores.
- Intensificación sobre el incumbente.
- Perturbaciones para escapar de óptimos locales.
- Fix-and-optimize o solución exacta parcial cuando ya exista soporte.

No implementes todos los operadores por rutina. Diagnostica el cuello de botella y elige los cambios con mayor posibilidad de reducir el costo de esa instancia.

Después de cada cambio:

1. Ejecuta las pruebas relevantes.
2. Repite una comparación controlada.
3. Conserva el cambio solo si mejora resultados o habilita una estrategia justificable.
4. No reemplaces una solución factible por una peor.
5. No sacrifiques factibilidad por costo.

## 5. Regla de incumbente

Mantén durante toda la tarea una única solución incumbente:

```text
incumbente = mejor solución factible encontrada hasta el momento
```

Actualízala solamente cuando:

```text
nuevo_costo < costo_incumbente
```

Cada vez que haya una mejora importante:

- Guarda una copia válida.
- Registra algoritmo, parámetros, semilla y costo.
- Evita perderla por ejecuciones posteriores.

## 6. Validación obligatoria

Una solución solo puede declararse ganadora si supera todas estas comprobaciones:

- Todos los clientes están atendidos.
- Cada cliente aparece exactamente una vez.
- Se respetan capacidades de vehículos.
- Se respetan capacidades de depósitos.
- Se respetan límites de vehículos o rutas.
- Los depósitos usados están abiertos correctamente.
- Las rutas comienzan y terminan según el formato oficial.
- El costo se recalcula desde cero.
- El costo escrito coincide con el costo recalculado.
- El checker oficial del repositorio la acepta.
- El archivo puede volver a parsearse.
- El costo es estrictamente menor que el objetivo rival.

Nunca declares éxito basándote solamente en el costo que imprimió un algoritmo.

## 7. Archivo de salida

No sobrescribas la instancia ni elimines soluciones previas.

Respeta primero la convención de carpetas existente. Si no hay una convención clara, guarda la mejor solución en:

```text
solutions/official/<instancia>_best.sol
```

Si ese nombre ya existe, reemplázalo únicamente cuando la nueva solución sea factible y estrictamente mejor.

También registra un archivo breve de resultados, si el repositorio ya posee una carpeta apropiada, con:

- Instancia.
- Objetivo rival.
- Mejor costo encontrado.
- Diferencia respecto del objetivo.
- Semilla.
- Algoritmo.
- Tiempo.
- Ruta de la solución.
- Resultado del checker.

## 8. Criterio de término

### Éxito

La tarea termina con estado `SUPERADO` cuando existe una solución validada con:

```text
mejor_costo_encontrado < mejor_solucion
```

### Objetivo no alcanzado

Si se agota el presupuesto sin superar el objetivo:

- Conserva la mejor solución factible encontrada.
- No afirmes que el objetivo fue superado.
- Reporta la brecha restante.
- Resume las configuraciones y mejoras probadas.
- Indica cuál sería el siguiente experimento de mayor valor.

Calcula:

```text
brecha_absoluta = mejor_costo_encontrado - mejor_solucion
brecha_porcentual = 100 * (mejor_costo_encontrado - mejor_solucion) / mejor_solucion
```

Una brecha negativa significa que el objetivo fue superado.

## 9. Respuesta final compacta

Usa este formato:

```text
ESTADO: SUPERADO | NO SUPERADO

Instancia:
Archivo de instancia:
Objetivo rival:
Mejor costo local inicial:
Mejor costo encontrado:
Diferencia:
Mejora frente al mejor local:
Algoritmo/configuración:
Semilla:
Tiempo total:
Checker:
Solución guardada:
Código modificado:
```

Si el estado es `SUPERADO`, destaca claramente:

```text
<mejor_costo_encontrado> < <mejor_solucion>
```

## Reglas del repositorio

- No hagas `git push`.
- No hagas `git reset --hard`.
- No borres resultados útiles.
- No modifiques archivos ajenos a la optimización.
- No expongas credenciales, tokens o enlaces privados.
- No uses la solución rival como si fuera un archivo disponible: normalmente solo se conoce su costo.
- No garantices de antemano que el objetivo será superado.
- Prioriza resultados verificables por sobre explicaciones extensas.
