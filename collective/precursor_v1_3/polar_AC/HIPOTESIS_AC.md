# ESTUDIOS A y C — Registro previo (escrito antes de ejecutar)
Fecha: 2026-09-18. Entorno E_commons-v2.2. Metapoblación de 6 grupos × 10 agentes, 25 generaciones
(5 de calentamiento), mutación simétrica 0,01. Medida principal: contención media de las generaciones 21–25.
SESOI = 0,10; α = 0,05. Potencia limitada: 8 semillas por celda (se declara de antemano).

## Estudio A — Igualación del beneficio privado (semillas 2500–2507)
Motivación: la imitación del más exitoso erosiona la contención porque los no contenidos sobreviven más.
Manipulación: en el paso de imitación, la aptitud individual de cada agente contenido se compensa con el
déficit medio observado en su grupo (aptitud ajustada = aptitud + déficit·rasgo). Solo cambia la señal que
usa la imitación; la ecología, la supervivencia real y la selección entre grupos quedan idénticas.
Celdas: 2 regímenes (imitación del éxito / conformidad) × 2 (con y sin igualación). Inicio p0 = 0,60.

H-A1: con imitación del éxito, la igualación aumenta la contención final (≥ SESOI, Mann-Whitney unilateral).
   F: diferencia < SESOI o no significativa.
H-A2 (interacción): el efecto de la igualación es mayor bajo imitación del éxito que bajo conformidad
   (diferencia de diferencias, IC95 % bootstrap que excluya cero).
   F: el intervalo incluye cero.
Predicción: ambas se cumplen; la conformidad depende de la frecuencia, no de la recompensa.

## Estudio C — Puente entre capas (semillas 2600–2607)
Pregunta: ¿la arquitectura de regulación local mueve la frontera poblacional, con transmisión y ecología
idénticas? Ningún modelo de masa crítica o de transmisión cultural lo predice.
Régimen fijo: conformidad + selección entre grupos. Inicio p0 = 0,50 (cerca de la frontera).
Variantes de la capa 1 (todo lo demás idéntico):
  BASE            arquitectura completa (tres críticos, γ largo 0,99, exploración dirigida ω = 0,5).
  SIN_ANCLA       se elimina el crítico de viabilidad propia del criterio de decisión.
  HORIZONTE_CORTO γ largo = 0,5.
  SIN_DIRIGIDA    ω = 0 (solo azar ciego).

H-C1: al menos una variante difiere de BASE en la contención final (≥ SESOI y p < 0,05, Mann-Whitney;
   corrección de Holm sobre las tres comparaciones).
   F: ninguna difiere → la capa 1 no afecta al destino poblacional y la extensión multinivel pierde su
   predicción distintiva en este entorno.
H-C2: SIN_ANCLA reduce la contención final respecto a BASE (dirección predicha).
   F: no la reduce.
Predicción: espero que SIN_ANCLA y HORIZONTE_CORTO reduzcan la contención (ambos degradan la conducta
sostenible), y que SIN_DIRIGIDA tenga poco efecto. Riesgo declarado: con la conformidad dominando la
dinámica, la capa 1 podría no mover nada, que es justamente lo que la prueba debe poder mostrar.
