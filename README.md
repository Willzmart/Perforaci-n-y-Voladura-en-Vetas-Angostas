# Voladura en vetas angostas: avance y producción

Web app en Streamlit para diseñar la perforación y voladura en **vetas angostas y labores artesanales subterráneas**. Cubre tres labores: avance en veta, avance en desmonte y producción en veta. Se basa en:

- **[C]** Carpio Chávez, A. (2025). *Diseño de perforación y voladura de labores de avance*. Cusco.
- **[LJ]** López Jimeno, C. et al. (1987). *Manual de perforación y voladura de rocas*. IGME, Madrid. Cap. 20, 22 y 24.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Avance en veta

- **Secciones** desde 1.0 × 1.5 m hasta 2.4 × 2.7 m. Debajo de 1.5 × 1.5 m la app avisa sobre sostenimiento, ventilación y evacuación.
- **Cueles para sección estrecha:**
  - Quemado simple, Sarrois, Sueco y Fagersta [LJ cap. 22].
  - Cilíndrico de 4 secciones, versión Carpio y versión LJ con desviación de perforación.
  - Cuele en V. Su avance queda limitado a 45–50 % del ancho [LJ].
- **Veta en el frente:** potencia, buzamiento, posición, ley y densidades. Se dibuja como una franja inclinada con el ancho aparente w / sen β.
- **"Ubicar el cuele en la veta":** traslada el cuele al eje de la veta, para que la cara libre quede en el mineral.
- **Voladura en dos etapas:**
  - Etapa 1: la veta junto con el cuele.
  - Etapa 2: el desquinche de la caja, que sale hacia la cara libre que dejó la veta.
  - La app compara la dilución, la ley diluida y el contenido metálico contra el disparo a frente completo.
  - Ejemplo con un frente de 1.8 × 2.1 m y una veta de 0.40 m a 70°: la dilución baja de ≈ 75 % a ≈ 9 %.
- **Equipos:** jackleg, stoper y martillo manual con empujador [LJ §6]. Los cartuchos de 7/8" son los predeterminados.

## Avance en desmonte

Para cruceros, accesos o ventanas hacia la veta, by-pass, cámaras de refugio y galerías exploratorias en roca estéril.

- **Mismos cueles, secciones y métodos de carga** que el avance en veta. El cuele va centrado por defecto; se puede desplazar hacia un hastial o cambiar su altura.
- **Contorno:** en labores permanentes la voladura controlada viene activada por defecto [LJ cap. 22]. Si se desactiva, la app lo advierte.
- **Acarreo:** sobreexcavación, esponjamiento, volumen esponjado y viajes por disparo (carretilla, carro U-35, U-60, scoop pequeño o capacidad propia).
- **Programa de avance:** a partir de la longitud de la labor y los disparos por guardia calcula disparos, días, explosivo total, desmonte total y costo de perforación y voladura. Incluye una curva de avance acumulado.
- **Indicadores por metro:** kg/m, m perforados/m y US$/m.

## Producción en veta

| Método | Base de cálculo |
|---|---|
| Corte y relleno ascendente: realce | LJ cap. 24 §6: barrenos a 50–65°, de 4 m como máximo, con 200–260 g/t |
| Almacenamiento provisional (shrinkage) | Usa la geometría de realce y extrae sólo el esponjamiento (ver limitaciones) |
| Corte y relleno ascendente: breasting | LJ cap. 24 §6, perforación horizontal |
| Taladros largos en vetas angostas (subniveles) | LJ cap. 24, más Langefors (cap. 20) |
| Subniveles con abanicos en veta angosta | LJ cap. 24: CE = CE₀ + 0.03 L + 0.40/AV |
| Cámaras y pilares en mantos angostos | LJ cap. 24, perforación frontal |

- **Burden:** se elige entre Langefors [LJ cap. 20], Holmberg y Pearse con Kv [C ec. 06].
- **Advertencias:** la app avisa si el ancho de minado queda bajo el ancho aparente de la veta, o bajo 0.8 m.

## Pestañas

- **Memoria de cálculo:** cada fase con su ecuación y la fuente. Incluye la fase "Veta y dilución".
- **Veta y dilución:** balance de mineral y desmonte, y comparación entre una y dos etapas.
- **Fragmentación:**
  - Kuz-Ram y una ruta ML con Evolución Diferencial para ajustar el burden según el P80 objetivo.
  - **% de sobretamaño** sobre la parrilla (por defecto ≤ 20 cm, DS 024-2016-EM).
- **Vista 2D:** frente, planta o sección con la franja de veta. Se colorea por zona o por etapa.
- **Vista 3D:**
  - Muestra taladros con taco, carga y cebo, y la línea de iniciación.
  - En taladros largos, el tajeo se inclina según el buzamiento.
- **Secuenciamiento:**
  - Métodos MILP, heurística o por filas.
  - Permite secuenciar el disparo completo o cada etapa por separado.
  - Detonadores Exsanel MS/LP, o carmex con una longitud mínima de mecha de 1.50 m.
  - Control de la carga máxima por retardo.
- **Resumen:** incluye toneladas, dilución, ley diluida y costo por tonelada de mineral. Se exporta en CSV y JSON.

## Referencias normativas (verificar texto vigente)

- DS 024-2016-EM, Art. 292 e): abertura de parrilla.
- DS 024-2016-EM, Art. 237 c): longitud mínima de la mecha de seguridad.

Los valores se usan como referencia en la app. Confirme el texto vigente y sus modificatorias antes de aplicarlos.

## Supuestos y limitaciones

- **Shrinkage:** los libros no desarrollan este método. Se usan las fórmulas de realce [LJ] y el balance de esponjamiento.
- **Franja de veta:** la veta se modela como una franja de potencia constante con contactos planos. No representa ramaleos, lazos ni acuñamientos.
- **Clasificación de taladros:** un taladro cuenta como "en veta" según la posición de su boca en el frente, con una tolerancia de 5 cm.
- **Sobrerrotura en dos etapas:** es un porcentaje del mineral que el usuario ingresa (10 % por defecto).
- **Geometrías aproximadas:** los cueles Fagersta y Sueco se toman de las figuras del libro.
- **Constantes referenciales:** las de vibración, combustión de mecha y costos.
- **Sustituto ML:** el modelo incluido para la ruta B es de demostración. Reemplácelo por el modelo entrenado de la tesis.
