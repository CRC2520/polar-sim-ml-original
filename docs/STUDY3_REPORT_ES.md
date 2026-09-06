# Estudio 3: evaluación externa de la red inter-polar

## Decisión científica

Se completó el siguiente paso experimental. La decisión registrada es
`no_confirmed_network_advantage` tanto en el estudio principal como en el panel
activo separado. La arquitectura por capas permanece como propuesta de investigación;
no se reduce al controlador ensayado ni se considera validada íntegramente.

Los resultados originales están fijados en el commit
`31eec6e0533d023579b747736a3bb5ef6b03ac7e`. Este informe es interpretación posterior,
no una modificación del protocolo, parámetros, datos o umbrales.

## Cronología verificable

Todos los horarios siguientes son UTC del 6 de septiembre de 2026.

| Registro o ejecución | Evidencia |
|---|---|
| Protocolo y código antes del desarrollo | `039bdb4031d8ae040983d2a9111e7f8959fe66cd` |
| Selección de desarrollo congelada, 04:25:09 | `e44f783e096aafa9822ce517994780d5a89c3ae0` |
| Panel activo registrado por separado, 04:31:29 | `054469956cf27abc193827956f5e5a4c63cbdd99` |
| Inicio del estudio final principal, 04:34:58 | `results_study3/final/STARTED.json` |
| Inicio del panel activo, 04:39:10 | `results_study3/active/final/STARTED.json` |

El panel activo se diseñó después de conocer la selección de desarrollo, pero
antes de generar cualquiera de los dos conjuntos finales. Se declara esa dependencia.
No es un cambio retroactivo del estudio principal ni una replicación externa.

## Diseño ejecutado

Desarrollo: 8.208 ejecuciones, seis semillas. Estudio principal: 5.760 ejecuciones,
40 semillas nuevas. Panel activo: 2.880 ejecuciones y otras 40 semillas nuevas.
En total son 8.640 ejecuciones finales y 552.960 transiciones. La unidad inferencial
es la semilla completa: 40 por panel, no 8.640 observaciones independientes.

Se cruzan dos familias sintéticas de efectos, tres topologías ambientales y dos
presupuestos. Las demandas, ganancias, ruido y otras entradas se mantienen iguales
al variar topología/presupuesto. El resultado principal es el MSE ponderado entre
los efectos observados y la demanda externa, no HGI/INC. Los objetivos están
completamente observados: no se evalúa aquí recuerdo episódico.

Cinco comparaciones direccionales usan bootstrap emparejado por semilla, 20.000
remuestreos y límites superiores unilaterales del 99 % (ajuste .05/5 dentro de cada
panel). Los umbrales mínimos se fijaron antes de los resultados: -0,001 para utilidad
de red/tensión y -0,0005 para topologías/característica alternativa. Los intervalos
percentiles son aproximados y condicionan en una sola selección de desarrollo.

## Resultado principal: la selección apaga la red

Las cuatro arquitecturas seleccionaron eta=1 y a=b=0. En consecuencia, el controlador
seleccionado, sus ablaciones y las topologías/características alternativas coinciden:
MSE 0,009743124. El comparador gradiente obtiene 0,010632588; el control inactivo,
0,229296706. La diferencia frente al gradiente corresponde al controlador base, no
puede atribuirse a una red que está apagada.

Los cinco contrastes principales son cero. Esto describe la selección dentro de la
rejilla definida; no demuestra equivalencia de todas las redes activas ni estima su
contribución causal. El diseño tiene una limitación explícita de identificabilidad:
cuando ambos coeficientes son cero, eliminar las vías o cambiar la topología no cambia
el comportamiento. El panel separado aborda ese límite sin borrar el resultado.

## Panel con tensión activa

Se eligió la mejor configuración ya evaluada en desarrollo restringiendo b!=0.
Full/rewired/reverse seleccionaron eta=1,a=0,b=-0,2; la característica genérica,
eta=1,a=0,b=+0,2. K está activa; W sigue apagada. No es una evaluación de aprendizaje
libre de aristas ni de propagación simultánea W+K.

| Condición | MSE externo | Costo normalizado | MSE tras cambios |
|---|---:|---:|---:|
| Red de tensión propuesta | 0,011273843 | 0,389478 | 0,018367472 |
| Misma política sin K | 0,010378349 | 0,408052 | 0,016245697 |
| Controlador sin red reajustado | 0,010378349 | 0,408052 | 0,016245697 |
| Topología reconectada | 0,011316721 | 0,389427 | 0,018405041 |
| Topología inversa | 0,011320120 | 0,389238 | 0,018418171 |
| Característica no lineal genérica | 0,013642145 | 0,430955 | 0,019310803 |

Full menos controlador sin red: +0,000895494; IC descriptivo 95 %
[+0,000803756; +0,000996343]. Es un incremento de error medio del 8,63 %, no una mejora.
El costo disminuye 4,55 %, pero el error tras cambios aumenta 13,06 %. No hubo
violaciones duras. Las guardas se cumplen porque son tolerancias: no exigen que
todos los resultados mejoren.

La bandera separada de deterioro práctico no se activa: el límite inferior 99 %
+0,000789023 no supera el umbral registrado +0,001. Eso NO significa que no se
observe deterioro; significa que no se supera esa regla específica.

La topología nominada no demuestra la ventaja mínima frente a las otras dos:
las diferencias son -0,000042879 y -0,000046277, con intervalos 95 % que incluyen cero.
Sí supera a la característica genérica ensayada: diferencia -0,002368302,
IC 95 % [-0,002690542; -0,002046348], límite superior 99 % -0,001988593.
Es el único contraste de superioridad que cumple el umbral. Ganar frente a esa
alternativa concreta no compensa perder frente al controlador sin red ni demuestra
superioridad frente a los controladores genéricos en general.

La intervención K cambia acciones hasta 0,388094426: la vía funciona causalmente.
Lo que no queda establecido es su utilidad neta bajo el criterio de desempeño.

## Verificación y conservación

92 pruebas aprobadas antes de la evaluación final. Se verificaron todos los registros,
hashes, configuraciones, conjuntos de ensayos, observaciones, efectos y métricas de
ambos paneles. Se reprodujeron exactamente las decisiones desde los archivos.
24 reproducciones completas de controladores del panel activo dieron diferencia
máxima de acciones igual a cero. Son verificaciones internas, no replicación externa.

Los Estudios 1/2 y el motor previo no se modificaron. La publicación en LaTeX continúa
en un único manuscrito que conserva teoría y arquitectura. Las funciones C/E/U,
IACL autónomo y conciencia no se infieren de los resultados de esta evaluación.

## Siguiente decisión de investigación

No promover esta configuración como mejora de rendimiento. Conservarla como mecanismo
ejecutable y evidencia negativa útil. Una explicación plausible es que la inhibición
compartida reduce esfuerzo e introduce subrespuesta; este estudio no aísla esa causa
frente a escala de coeficientes, proxy de tensión o error del estimador diagonal.

El próximo trabajo debe ser un diagnóstico exploratorio de esas contribuciones,
seguido de una hipótesis revisada y otro registro previo. La rejilla solo ensayó
magnitudes no nulas de 0,2: ganancias menores/continuas, aristas individuales aprendidas,
compuertas contextuales y otras familias de tareas siguen sin evaluar. Nada justifica
concluir que toda red inter-polar sea inviable, ni afirmar que la arquitectura completa
ya aporta conciencia o ventaja exclusiva.

## Reproducir los informes existentes

En la rama que contiene los datos, con el entorno registrado:

```bash
python -m unittest discover -s tests -v
python -m study3.run regenerate
python scripts/run_study3_active.py regenerate
```

Las trazas completas están en los TAR de `results_study3/final/` y
`results_study3/active/final/`. El paquete liviano de resultados incluye informes,
protocolos, manifiestos y código; no duplica las trazas completas. Las semillas
finales ya están abiertas: reproducirlas no es una nueva prueba confirmatoria.
