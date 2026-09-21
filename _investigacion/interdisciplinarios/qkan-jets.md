---
layout: proyecto-layout
title: "Quantum Sine-Kolmogorov-Arnold Networks for High Energy Physics Analysis"
proyecto: "Quantum KAN & Jets"
seccion: "Resumen técnico"
area: "Interdisciplinarios"
tipo: "macro"
estatus: "Completado"
description: "Redes de Kolmogorov-Arnold cuánticas para etiquetado de jets, mi proyecto de Google Summer of Code 2026 con ML4SCI."
orden: 1
---
# Redes Kolmogorov-Arnold cuánticas para etiquetado de jets de quark top

*Un resumen técnico del proyecto QKAN, Google Summer of Code 2026 en ML4SCI*

Este proyecto fue desarrollado por Jorge Toral y puede encontrarse en este [repositorio de GitHub](https://github.com/Jorge-1501/QKANs-ML4SCI_2026)

---

## 1. El problema y la idea central

En el Gran Colisionador de Hadrones, un quark top se desintegra casi instantáneamente y produce un **jet** (un chorro de partículas colimado) con una subestructura interna característica, distinta de la de un jet ordinario originado por un quark ligero o un gluon (QCD). Distinguir estos dos tipos de jet, **top tagging**, es un problema de clasificación binaria bien estudiado, con arquitecturas clásicas de referencia que alcanzan AUCs superiores a 0.96-0.98 en el dataset público que usamos en este proyecto ([Kasieczka et al., 2019](#ref-kasieczka-2019)): jets simulados a 14 TeV con Pythia8 y una tarjeta de detector Delphes tipo ATLAS, reconstruidos con anti-$k_T$ y $R=0.8$ en el rango $p_T \in [550, 650]$ GeV.

Este proyecto no compite por superar esas arquitecturas. Su pregunta es distinta: **¿puede una red neuronal cuántica variacional (VQC) hacer esta tarea, y qué papel puede jugar una red clásica en hacerla viable?**

La limitación de fondo es simple: el costo de simular un circuito cuántico crece como $O(2^Q)$ con el número de qubits $Q$. Alimentar directamente un jet con docenas de variables a un VQC es impráctico. La solución que exploramos usa una **red Kolmogorov-Arnold (KAN)** clásica como preprocesador y filtro de qubits: se entrena, se poda hasta una topología pequeña e interpretable, y esa topología podada, no un diseño arbitrario, es la que decide cuántos qubits necesita el circuito cuántico y cómo se conectan. El modelo clásico determina el recurso cuántico, en lugar de dejarlo a prueba y error.

El flujo completo, documentado en tres notebooks (`EDA_top.ipynb`, `Training_process.ipynb`, `Results.ipynb`), preprocesa los jets en réplicas balanceadas, entrena y poda la KAN, extrae cada arista sobreviviente a una base compacta (Chebyshev o seno) como *warm start*, afina la QKAN en simuladores ideal, con ruido por disparos finitos (*shots*) y con ruido de hardware, y compara todo contra un Random Forest clásico.

## 2. ¿Por qué una KAN y no un MLP?

Las redes Kolmogorov-Arnold, propuestas por [Liu et al. (2024)](#ref-liu-2024-kan) partiendo del teorema de representación de Kolmogorov-Arnold (KAT), invierten el diseño de un perceptrón multicapa tradicional. En un MLP, las funciones de activación son fijas (ReLU, tanh, ...) y viven en los nodos; los pesos, que sí se aprenden, son escalares en las aristas. En una KAN es al revés: **cada arista lleva su propia función univariada aprendible**, parametrizada como un spline, y los nodos únicamente suman. El teorema KAT garantiza que cualquier función continua multivariada puede escribirse como

$$f(x_1, \dots, x_n) = \sum_{q=1}^{2n+1} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

es decir, como una composición de funciones univariadas y sumas. Esto tiene dos consecuencias que explotamos directamente. Primero, **interpretabilidad**: cada arista, al terminar el entrenamiento, puede ajustarse simbólicamente contra una librería de funciones candidatas, dando una fórmula cerrada para lo que esa arista «hace». Segundo, y crucial para este proyecto, **podabilidad estructurada**: como cada arista es una unidad funcional independiente, se le puede medir una atribución (cuánto contribuye a la salida) y eliminarla si es despreciable, sin romper la interpretación del resto de la red.

Un trabajo posterior de los mismos autores, [KAN 2.0 (Liu et al., 2024)](#ref-liu-2024-kan2), introduce además los **nodos de multiplicación**: el teorema KAT en su forma clásica solo garantiza composición mediante sumas, pero dejar que algunos nodos ocultos multipliquen sus entradas en vez de sumarlas permite expresar interacciones entre variables de forma directa. Esta idea es la que adoptamos: la arquitectura clásica de este proyecto (`HEPKAN`, una subclase de [pykan](https://github.com/kindxiaoming/pykan)) usa una capa oculta con nodos de suma *y* de multiplicación en paralelo.

## 3. Explorando los datos: qué hace a un jet de top distinto

Antes de entrenar nada, el notebook de EDA caracteriza el dataset. El archivo HDF5 no documenta el orden de sus columnas, así que lo inferimos de los datos: cada bloque de cuatro columnas corresponde a una partícula constituyente del jet, con la energía en la primera posición y las tres componentes de momento después, un patrón que se reconoce en las medias de columna, mucho más grandes en la primera de cada bloque de cuatro.

De los histogramas y perfiles radiales surgieron cuatro observaciones que guiaron todo el diseño posterior:

- **La masa invariante discrimina, pero no basta.** $m_{jet} = \sqrt{E^2 - p_x^2 - p_y^2 - p_z^2}$ tiene un pico claro en la señal alrededor de la masa del quark top (~173 GeV), mientras que el fondo es más ancho. Las dos distribuciones se solapan por secciones, así que seleccionamos la ventana **145-205 GeV** para los experimentos principales: dentro de ella, la pista trivial de la masa queda en buena medida removida y los clasificadores deben apoyarse en información más sutil.
- **Los jets de top son más poblados**: la multiplicidad (número de constituyentes) es sistemáticamente mayor y más dispersa que en QCD, consistente con una desintegración en tres cuerpos, para este intervalo seleccionado de masa.
- **Los jets de top son más difusos**: el perfil radial de energía y la fracción acumulada de $p_T$ crecen más gradualmente en tops que en QCD, donde la energía está más concentrada cerca del eje del jet.
- Los diagramas de dispersión $\eta$-$\phi$ no fueron concluyentes por ruido y escala.

También detectamos un artefacto: la distribución de $\Delta R$ no muestra el corte nítido en 0.8 que debería, por una constante pequeña añadida para evitar división por cero. Filtramos los constituyentes fuera del radio del jet para corregirlo.

A partir de esto construimos la representación de entrada, combinando variables **globales** (masa del jet y multiplicidad, contando solo constituyentes con energía por encima de $10^{-8}$) con variables **locales** por cada constituyente retenido ($\Delta R$ al eje del jet y $p_{T,\text{rel}}$). Usando el criterio del 80% de la distribución acumulada de $p_T$, estimamos que unos 15 constituyentes bastarían para capturar la mayor parte de la energía; la canalización final, sin embargo, usa solo **los 10 constituyentes más energéticos de cada jet**, para un total de **22 entradas** (2 globales + 10 pares de variables locales). Cada jet original trae hasta 200 constituyentes almacenados; el corte a 10 es una decisión deliberada de compresión que retomamos en resultados y conclusiones, porque el modelo nunca ve el 95% de las partículas que el detector registró.

Las variables locales ya viven en $(0,1)$; a las globales les aplicamos una transformación logarítmica seguida de normalización tanh, conservando deliberadamente los valores atípicos porque los centros de las dos distribuciones de clase son similares y es la cola la que lleva información discriminante.

<figure>
  <img src="/assets/img/investigacion/qkan/dispersion.png" alt="Dispersion">
  <figcaption>Dispersión de los constituyentes de los jets en señales de top y QCD. Rango antes de cualquier corte de masa.</figcaption>
</figure>

<figure>
  <img src="/assets/img/investigacion/qkan/Invariant_mass.png" alt="Invariant Mass">
  <figcaption>Masa invariante de los jets en señales de top y QCD. Rango antes de cualquier corte de masa. </figcaption>
</figure>

Finalmente, como el número de eventos en la ventana de masa difiere entre clases (hay más tops), submuestreamos la clase mayoritaria para balancear, y dividimos el resultado en **5 subconjuntos disjuntos y balanceados por clase**. Cada semilla selecciona uno (`seed % 5`), de modo que varias semillas dan réplicas independientes de principio a fin en lugar de un único punto de estimación.

## 4. La arquitectura clásica y su poda

El modelo clásico tiene arquitectura `[22, [9, 9], 1]`: 22 entradas, una capa oculta con hasta 9 nodos de suma y 9 de multiplicación en paralelo, y una salida escalar. Los splines B usan grado $k=3$ y tamaño de malla (*grid*) 5, para un total de **8,568 parámetros**. En la corrida de referencia (seed 10), el modelo base alcanzó un AUC de prueba de **0.792**, deteniéndose temprano alrededor de la época 20 de las 60 planeadas.

`HEPKAN` introduce tres modificaciones prácticas sobre `pykan`, motivadas por los recursos computacionales limitados del proyecto: una corrección de un error en `prune_input` (pasaba un módulo en vez de un nombre de cadena al reconstruir el modelo podado, rompiendo la serialización de checkpoints), una rutina de graficado que reutiliza una sola figura de Matplotlib para todas las aristas y omite las ya podadas, y un registro de historial no-op que evita escribir un checkpoint en cada mutación del modelo durante poda y búsqueda simbólica.

La poda combina un umbral de entrada de 0.01 con umbrales de atribución de 0.04 (nodos) y 0.06 (aristas), y añade una restricción dura pensada para el límite cuántico: un **tope de fan-in de dos**, que conserva solo las dos aristas de entrada más fuertes por neurona oculta. Tras podar, la estructura se reentrena 20 épocas, durante las cuales el AUC de validación sube de 0.714 a ~0.754.

El resultado, para la corrida de referencia, es sorprendentemente compacto: de 22 variables de entrada, sobreviven solo **2** (masa del jet y multiplicidad), organizadas en **1 nodo de suma y 5 nodos de multiplicación**. Este hallazgo retroalimenta la importancia de variables (MDI/Gini) del Random Forest de referencia (sección 6), que de forma independiente también señala a la masa y la multiplicidad como las más predictivas entre las 22 disponibles: una confirmación cruzada de que la poda agresiva no descarta señal relevante.

El notebook también corre un ajuste simbólico sobre cada arista sobreviviente, emparejándola contra una librería de funciones candidatas para obtener una interpretación en forma de fórmula: la principal ventaja de interpretabilidad de las KANs, y parte de la rama puramente clásica del proyecto. La rama cuántica, en cambio, no usa esta fórmula simbólica sino la **respuesta numérica** de cada arista.

## 5. Del grafo clásico al circuito cuántico

Un extractor aísla la respuesta de cada arista activa desconectando las otras entradas de su nodo destino, y exporta un grafo serializado de nodos de suma y multiplicación. Para la corrida de referencia, ese grafo tiene **11 qubits**, 18 aristas de entrada, 5 transferencias `IsingZZ` y 6 aristas de salida. Es importante notar que los qubits se cuentan **por nodo acumulador sobreviviente, no por entrada**: la misma variable de entrada puede recargarse (re-uploading) en varios cables si alimenta a varios nodos ocultos distintos.

El circuito sigue cinco principios de diseño:

1. **Data re-uploading**: cada función univariada de una arista se modela mediante rotaciones repetidas del dato de entrada, en lugar de codificarlo una sola vez.
2. La capa oculta clásica decide qué variables importan y cuántos qubits se usan, el circuito hereda la topología.
3. **La suma es gratuita**: rotaciones $R_Z$ consecutivas en el mismo cable acumulan sus ángulos, así que los nodos de suma no requieren ninguna puerta de dos qubits.
4. La **multiplicación** se implementa con una puerta `IsingZZ` combinada con un `CNOT`.
5. Toda la información colapsa en un solo cable de salida, y la predicción es el valor esperado de Pauli-Z de un único qubit.

<figure>
  <img src="/assets/img/investigacion/qkan/retrained_model.png" alt="KAN pruned">
  <figcaption>Modelo KAN podado y reentrenado.</figcaption>
</figure>

<figure>
  <img src="/assets/img/investigacion/qkan/quantum-circuit.png" alt="Quantum Circuit">
  <figcaption>Circuito cuántico correspondiente al grafo clásico podado.</figcaption>
</figure>

La etapa de oculta-a-salida es un lector variacional, no una segunda capa literal de KAN, porque el valor de un nodo oculto vive en la fase de un qubit y no puede volver a cargarse sin una medición intermedia. Por esta razón, **solo se soportan redes de profundidad 2**, una limitación de diseño explícita, que dejamos como trabajo futuro.

El modelo puede correr en tres simuladores: `ideal` (`lightning.qubit`), `shots` (`default.qubit` con número finito de disparos) y `noisy` (Qiskit Aer con un modelo de ruido derivado de `FakeManilaV2`). El entrenamiento usa entropía cruzada binaria con logits, el optimizador Adam y un *scheduler* `ReduceLROnPlateau` que reduce la tasa de aprendizaje a la mitad cuando la validación se estanca. Como la simulación es costosa, cada época entrena sobre un subconjunto aleatorio fresco (~1,000 muestras) y valida sobre un subconjunto fijo.

## 6. Tres formas de inicializar el circuito: el warm start

Antes de afinar el circuito con descenso de gradiente, hay que decidir con qué ángulos empezar. Comparamos tres estrategias.

**Chebyshev**, siguiendo el diseño de la [Chebyshev-KAN (Sidharth et al., 2024)](#ref-sidharth-2024). Cada respuesta de arista aislada se ajusta como $y \approx \sum_{i=0}^{N} c_i T_i(x)$ sobre $[-1,1]$, y los coeficientes resultantes se convierten en los ángulos iniciales de rotación. El grado se fija en $N=4$ para todas las aristas. Esta decisión, de grado fijo en vez de adaptativo, viene de un error real diagnosticado durante el proyecto: originalmente buscábamos el grado más pequeño que superara un umbral de $R^2$, pero ese criterio casi siempre elegía grados bajos y producía métricas inconsistentes, porque el circuito perdía la información de la estructura clásica. El síntoma fue una caída abrupta del AUC de referencia (de ~0.80 a 0.26-0.36) cuando el conjunto de entrenamiento se hacía más pequeño; volver a un grado fijo restauró el comportamiento esperado.

**Base de seno.** Como alternativa implementamos una base sinusoidal de frecuencia fija, siguiendo el diseño de [**SineKAN** (Reinhardt et al., 2024)](#ref-reinhardt-2024): $y \approx \sum_k A_k \sin(\text{freq}_k \, x + \text{phase}_k)$, con las amplitudes $A_k$ obtenidas por mínimos cuadrados sobre una malla de frecuencias y fases *fija*, no aprendida. En SineKAN, la noción de «grado» de la arista corresponde exactamente al número de términos seno que se suman en esa malla: no hay un polinomio de grado creciente como en Chebyshev, sino más armónicos sinusoidales acumulados. Esta base es, además, la que usa Ria Khatoniar en la rama de lectura clásica de su proyecto de GSoC 2025 ([Khatoniar, 2025a](#ref-khatoniar-2025a), sección 8), y el script de referencia que usamos para portar fielmente la construcción de la malla de frecuencias y fases (constantes $A=0.9724$, $K=0.9884$, $C=0.9994$ del `SineKANLayer` original) proviene directamente de su código.

Con esta base fija replicamos el experimento de Chebyshev a pequeña escala, ajustando aristas reales extraídas del pipeline con ambas bases y comparando su $R^2$. El resultado confirma cuantitativamente lo que la teoría sugiere: el ajuste de la base de seno es moderado, con un **$R^2$ medio de aproximadamente 0.49-0.61**, muy por debajo del ajuste casi perfecto de Chebyshev (cercano a 1). Sin un término constante, la base de seno no puede representar desplazamientos estáticos (offsets) y produce $R^2$ negativo en algunas aristas. Añadir un término constante a la base es, con esta evidencia, el siguiente paso obvio.

Vale la pena situar este hallazgo frente a un artículo teórico más reciente sobre esta misma base: ["Sinusoidal Approximation Theorem for KANs" (Gleyzer et al., 2025)](#ref-gleyzer-2025) da una prueba constructiva de aproximación universal para KANs con base sinusoidal, en la línea del KAT original. Esto no contradice lo observado: garantiza que, con suficientes términos y libertad para ajustar frecuencia y fase, una base de senos *puede* aproximar cualquier función continua; nuestro resultado muestra que una malla fija, sin término constante y con solo $N=4$ armónicos, todavía no explota esa capacidad.

**Inicialización aleatoria.** Como control, conservamos exactamente la topología podada (mismos qubits y conexiones) pero muestreamos cada ángulo de $\mathcal{N}(0,1)$. Esto aísla el valor del conocimiento clásico transferido del valor de la topología por sí sola.

## 7. La línea base clásica: Random Forest

Para calibrar qué tan lejos está el enfoque cuántico de lo alcanzable clásicamente, añadimos un Random Forest de 500 árboles con pesos de clase balanceados. A diferencia de la canalización KAN, recibe las **22 variables completas**, sin poda. Usa las mismas claves de métricas que los entrenadores de la KAN, de modo que todos los modelos se agregan en una sola tabla de resultados.

## 8. Resultados

Todas las métricas por corrida se recolectan en una única tabla Parquet (74 filas a través de 6 semillas distintas). Analizamos dos regímenes.

### 8.1 Con corte de masa, cinco réplicas (semillas 10-14)

La siguiente figura resume el AUC de prueba medio y su desviación estándar sobre 5 semillas, para toda la cadena de modelos, del Random Forest hasta la QKAN inicializada al azar sin entrenar:

<figure>
  <img src="/assets/img/investigacion/qkan/auc_mass_cut.png" alt="AUC por modelo, régimen con corte de masa">
  <figcaption>AUC por modelo, régimen con corte de masa.</figcaption>
</figure>


Tres observaciones se desprenden de esta tabla. Primero, podar y simplificar simbólicamente cuesta ~0.03 de AUC respecto a la KAN base, y la QKAN entrenada queda un 0.02 adicional por debajo de la KAN reentrenada: el circuito alcanza AUC ~0.73-0.74 usando solo dos variables y 11 qubits. Segundo, **el afinado sí importa**: entrenar el circuito eleva el warm start de Chebyshev de 0.698 a 0.736 en el simulador ideal. Tercero, los tres backends (ideal, disparos finitos y ruidoso) difieren entre sí en no más de ~0.006 de AUC; dentro de nuestro modelo de ruido, el circuito no se degrada visiblemente.

La comparación entre bases de warm start confirma el orden Chebyshev > Seno > Aleatorio, tanto en AUC como en rechazo de fondo. A un punto de trabajo de eficiencia de señal del 50% ($\varepsilon_S = 0.5$), medimos:

<figure>
  <img src="/assets/img/investigacion/qkan/bkg_rejection_warmstart.png" alt="Rechazo de fondo por base de warm start">
  <figcaption>Rechazo de fondo por base de warm start.</figcaption>
</figure>

Es decir, inicializar el circuito con la base de Chebyshev, sin entrenamiento adicional, rechaza aproximadamente 3 veces más fondo que la inicialización aleatoria a la misma eficiencia de señal, y la base de seno queda en un punto intermedio, consistente con su ajuste de arista más débil.

Para verificar que estas diferencias no son ruido estadístico sobre solo cinco semillas, corrimos pruebas $t$ pareadas ($\alpha=0.05$), con hipótesis nula de que la inicialización aleatoria y cada warm start dan la misma exactitud y AUC:

| Comparación | Exactitud | AUC |
|---|---|---|
| Aleatorio vs Chebyshev | $t=-4.86$, $p=0.0082$ | $t=-20.17$, $p=3.6\times10^{-5}$ |
| Aleatorio vs Seno | $t=-5.00$, $p=0.0075$ | $t=-17.95$, $p=5.7\times10^{-5}$ |

Ambas hipótesis nulas se rechazan con claridad, aunque leemos este resultado con cautela porque descansa en solo cinco semillas.

Un punto adicional: la exactitud del circuito cuántico ronda solo 0.56-0.57, frente a ~0.71 de la KAN clásica. La matriz de confusión de referencia colapsa hacia la clase positiva (recall 0.98, precisión 0.53). La señal de ordenamiento, medida por AUC, sobrevive, pero el umbral fijo en 0.5 está mal calibrado para la salida del circuito, por eso reportamos AUC, y no exactitud, como métrica principal.

### 8.2 Casi el dataset completo, sin corte de masa

En un segundo régimen usamos prácticamente todos los eventos disponibles (con los mismos 10 constituyentes por jet, pero sin restringir la ventana de masa), en un solo bloque, semilla 42. Al no haber réplicas aquí, no podemos calcular barras de error ni pruebas de hipótesis: el resultado debe leerse como **una sola corrida**.

<figure>
  <img src="/assets/img/investigacion/qkan/auc_full_dataset.png" alt="AUC por modelo, régimen sin corte de masa">
  <figcaption>AUC por modelo, régimen sin corte de masa.</figcaption>
</figure>

Sin el corte de masa, los clasificadores pueden explotar directamente la masa del jet, así que todos los modelos alcanzan su discriminación más alta, y la QKAN entrenada mantiene un AUC superior a 0.90, notablemente más alto que en el régimen con corte, aunque, otra vez, su exactitud es más baja (0.72-0.74, recall 0.97, precisión ~0.65), repitiendo el mismo problema de calibración observado antes.

## 9. ¿Qué tan lejos estamos de la literatura?

Es natural preguntar cómo se comparan estos números contra el estado del arte publicado para el mismo dataset. El notebook comparativo más citado para este benchmark es el de [SebastianMacaluso/TopTagComparison](#ref-macaluso), que reúne 14 taggers clásicos y de aprendizaje profundo (ParticleNet, TreeNiN, ResNeXt, PFN, CNN, NSub, LBN, P-CNN, LoLa, EFN, EFP, TopoDNN, entre otros) sobre el dataset completo de 404 mil eventos, sin restricción de masa, con AUCs entre 0.967 y 0.985 y rechazo de fondo a $\varepsilon_S=0.3$ entre 295.2 (TopoDNN) y 1298.5 (ParticleNet).

Esta comparación exige una advertencia explícita: **esos números usan el dataset completo, sin cortes**, mientras que la mayoría de nuestros resultados con réplicas y barras de error corresponden al régimen con corte de masa agresivo (145-205 GeV), diseñado para remover la pista más fácil y forzar a los modelos a usar subestructura. No son, por tanto, directamente comparables punto por punto. El terreno común más razonable es nuestro régimen sin corte de masa (sección 8.2): ahí, la KAN clásica base alcanza AUC 0.959 y el Random Forest 0.965, en el vecindario del extremo inferior de la literatura, aunque todavía por debajo de arquitecturas especializadas como ParticleNet. La QKAN entrenada en este régimen llega a AUC 0.904, notable para un circuito de 11 qubits derivado de solo dos variables, pero claramente por debajo tanto de nuestra KAN clásica como de los taggers de la literatura.

Dos factores adicionales, más allá del corte de masa, amplían esta brecha. El primero es la compresión de variables: pasamos de 22 a solo 2 tras la poda, mientras que arquitecturas como ParticleNet o IAFormer consumen la nube completa de constituyentes con grafos o atención dispersa diseñados para esa alta dimensión. El segundo, evidente solo al revisar el preprocesamiento después de tener los resultados principales, es que **cada jet trae hasta 200 constituyentes y nuestra canalización solo usa los 10 más energéticos**. Es una simplificación deliberada para mantener el circuito simulable, pero deja fuera casi toda la subestructura de baja energía, precisamente donde arquitecturas como [IAFormer (Esmail et al., 2026)](#ref-esmail-2026) o [L-GATr (Brehmer et al., 2025)](#ref-brehmer-2025), equivariante de Lorentz, reportan ganancias. Es, junto con el corte de masa y la compresión a dos variables, un tercer motivo legítimo para la brecha frente al estado del arte.

## 10. Trabajo relacionado: otros enfoques de KAN cuántica

Este proyecto no es el único esfuerzo en combinar KANs con computación cuántica, y vale la pena situarlo frente a otros dos.

**El proyecto QKAN de Ria Khatoniar ([Khatoniar, 2025a](#ref-khatoniar-2025a), [2025b](#ref-khatoniar-2025b); GSoC 2025, también en ML4SCI)** es el antecedente más cercano y el que inspiró directamente la elección de la base de seno en este trabajo. Su primer informe describe un QKAN híbrido que combina codificación QSVT, combinación lineal de unitarios cuántica (LCU) y una prueba de Hadamard para la suma, con lectura clásica tipo KAN/SineKAN; reporta explícitamente que la arquitectura no pudo escalar más allá de cierto punto al dataset de quark-gluon, por fallos de memoria (*kernel crashes*) del simulador. Su segundo informe presenta una KAN completamente cuántica basada en una Quantum Circuit Born Machine (QCBM), con qubits de etiqueta y de posición, una capa de entrelazamiento (*LabelMixer*) y lectura vía Pauli-Z/X; ahí declara que extender el enfoque a quark-gluon o a predicción de masa de jet "todavía no pudo lograrse, principalmente porque la arquitectura actual es computacionalmente lenta", limitándose a simuladores. Ambos informes concluyen el programa señalando la intención de continuar a futuro. Este proyecto retoma esa dirección en la elección de la base sinusoidal para el warm start, usando como referencia directa el código de `SineKANLayer` de su propio repositorio.

**[QuKAN (Werner et al., 2025)](#ref-werner-2025)** explora un enfoque distinto: en vez de destilar una KAN clásica ya entrenada en un circuito, usa directamente una Quantum Circuit Born Machine como mecanismo generativo para las funciones univariadas de la KAN, desde el principio cuántico.

Los tres proyectos juntos dejan un patrón claro: escalar una KAN cuántica más allá de un par de variables es, en 2025-2026, un problema abierto compartido, ya sea por memoria de simulación ([Khatoniar, 2025a](#ref-khatoniar-2025a), [2025b](#ref-khatoniar-2025b)), por el costo de un mecanismo generativo cuántico ([Werner et al., 2025](#ref-werner-2025)), o, aquí, por la necesidad de podar agresivamente antes de poder simular el circuito.

## 11. Conclusiones

Cinco conclusiones resumen el proyecto.

1. **La poda clásica funciona como filtro de qubits.** La KAN reduce 22 entradas a dos variables y 11 qubits, y el circuito todavía alcanza un AUC de ~0.73 (con corte de masa) y ~0.90 (sin corte).
2. **El warm start transporta información real.** El orden Chebyshev > Seno > Aleatorio es consistente entre semillas y estadísticamente significativo sobre cinco réplicas. La inicialización aleatoria rinde a nivel de azar (AUC ~0.49), lo que muestra que la topología por sí sola no basta. La base de Chebyshev es mejor porque su ajuste por arista está mucho más cerca de la respuesta clásica real ($R^2$ cercano a 1, frente a 0.49-0.61 para la base de seno).
3. **El ruido simulado tiene un efecto pequeño.** La brecha entre los backends `ideal` y `noisy` es de a lo sumo ~0.006 de AUC en el régimen con corte de masa, y de ~0.001 tras el entrenamiento en el régimen sin corte. Es alentador, pero se trata de un modelo de ruido simulado, **no de un resultado en hardware real**.
4. **El corte de masa es el problema más difícil.** Con él, el AUC cae de ~0.96 a 0.82 para el Random Forest, y de 0.90 a 0.73 para la QKAN: remover la pista simple de la masa obliga a los modelos a apoyarse en subestructura, que solo dos variables sobrevivientes pueden capturar parcialmente.
5. **El techo clásico lo marca el Random Forest.** Tiene acceso a las 22 variables y lidera en ambos regímenes; el modelo cuántico no lo supera. El valor del enfoque híbrido está en producir un circuito compacto e interpretable que preserva una fracción considerable de ese desempeño, no en una ganancia de exactitud sobre lo clásico.

## 12. Contribuciones

- Una canalización reproducible y consciente del régimen de datos: los distintos regímenes están codificados en la organización de directorios, los subconjuntos de réplicas están balanceados por clase, cada etapa es idempotente vía checkpoints, y los resultados de varias semillas se recolectan en una sola tabla Parquet.
- `HEPKAN`, una subclase de `pykan` que corrige un error de serialización en la poda de entradas y reduce la sobrecarga de graficado y registro.
- Una regla de poda diseñada específicamente para los límites cuánticos, combinando umbrales de atribución con un tope duro de fan-in.
- Un puente clásico-a-cuántico: el extractor convierte una KAN podada en un grafo de suma/multiplicación, y el constructor de la QKAN convierte ese grafo en un circuito de PennyLane.
- Dos bases intercambiables más un control, que permiten medir el valor del warm start en tres backends de simulación, evaluados antes y después del entrenamiento.
- Una falla diagnosticada y corregida: la búsqueda adaptativa de grado de Chebyshev colapsaba silenciosamente el AUC de referencia en conjuntos de datos más pequeños; encontramos la causa y la reemplazamos por un grado fijo.
- Un benchmark clásico, el Random Forest, evaluado con las mismas métricas y la misma partición de datos.
- Un análisis exploratorio que documenta la disposición del dataset, el artefacto de $\Delta R$ y la elección de 10 constituyentes y 22 entradas.

## 13. Limitaciones y trabajo futuro

- **Pocas réplicas**: las pruebas estadísticas usan cinco semillas, y el régimen sin corte de masa es una sola corrida sin incertidumbre estimada.
- **Compresión extrema**: la poda deja ~2 variables de entrada; relajarla eleva el número de qubits más allá de lo simulable en tiempo razonable.
- **Solo los 10 constituyentes más energéticos de hasta 200 por jet**, restricción adicional que probablemente descarta subestructura de baja energía relevante.
- **Calibración**: exactitud y precisión cuánticas son débiles incluso donde el AUC es bueno; falta estudiar un umbral de decisión afinado.
- **Base de seno**: necesita un término constante o más frecuencias antes de poder juzgarse de forma justa frente a Chebyshev.
- **Solo ruido simulado**: `FakeManilaV2` aproxima un dispositivo real, pero no lo es.
- **Solo redes de profundidad 2**: KANs más profundas requerirían medición intermedia y recodificación.
- **Otros datasets**: existe una canalización de preprocesamiento para etiquetado de quark-gluon, y la detección de Higgs solo se menciona como referencia futura, sin desarrollo.

En suma, el proyecto muestra que una KAN clásica puede decidir la forma de un circuito cuántico, que el conocimiento transferido a través de una buena base importa, y puede medirse, no solo intuirse; y que el modelo compacto resultante preserva una parte útil de la señal de clasificación. Todavía no iguala la mejor línea base clásica, y ese límite se reporta junto con los resultados.

## Agradecimientos

Quiero agradecer a mi amigo Eduardo Villamil por proporcionarme recursos computacionales para este proyecto, a ML4SCI por la oportunidad de participar en Google Summer of Code 2026, y a todos los administradores, mentores y miembros del programa por su retroalimentación útil a lo largo del proyecto.

---

## Referencias

- <a id="ref-brehmer-2025"></a>Brehmer, J., Bresó, V., de Haan, P., Plehn, T., Qu, H., Spinner, J., & Thaler, J. (2025). *A Lorentz-equivariant transformer for all of the LHC*. SciPost Physics. https://arxiv.org/abs/2411.00446
- <a id="ref-esmail-2026"></a>Esmail, W., Hammad, A., & Nojiri, M. (2026). IAFormer: Interaction-aware transformer network for collider data analysis. *SciPost Physics, 20*, Article 108. https://arxiv.org/abs/2505.03258
- <a id="ref-gleyzer-2025"></a>Gleyzer, S., Nguyen, H., Ramakrishnan, D. P., & Reinhardt, E. A. F. (2025). Sinusoidal approximation theorem for Kolmogorov–Arnold networks. *Mathematics, 13*(19), Article 3157. https://doi.org/10.3390/math13193157
- <a id="ref-kasieczka-2019"></a>Kasieczka, G., Plehn, T., Thompson, J., & Russell, M. (2019). *Top quark tagging reference dataset* (Version v0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2603256
- <a id="ref-khatoniar-2025a"></a>Khatoniar, R. (2025a). *GSoC 2025 \| Quantum Kolmogorov-Arnold networks for high energy physics analysis at the LHC (parte I)* [Entrada de blog]. Medium. https://medium.com/@riakhatoniar1234/gsoc-2025-quantum-kolmogorov-arnold-networks-for-high-energy-physics-analysis-at-the-lhc-a98207bf6d4c
- <a id="ref-khatoniar-2025b"></a>Khatoniar, R. (2025b). *GSoC 2025 \| Quantum Kolmogorov-Arnold networks for high energy physics analysis at the LHC (parte II)* [Entrada de blog]. Medium. https://medium.com/@riakhatoniar1234/gsoc-2025-quantum-kolmogorov-arnold-networks-for-high-energy-physics-analysis-at-the-lhc-part-8b44f5616e6f
- <a id="ref-liu-2024-kan"></a>Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). *KAN: Kolmogorov-Arnold networks*. arXiv. https://arxiv.org/abs/2404.19756
- <a id="ref-liu-2024-kan2"></a>Liu, Z., Ma, P., Wang, Y., Matusik, W., & Tegmark, M. (2024). *KAN 2.0: Kolmogorov-Arnold networks meet science*. arXiv. https://arxiv.org/abs/2408.10205
- <a id="ref-macaluso"></a>Macaluso, S. (n.d.). *TopTagComparison* [Repositorio de código]. GitHub. Recuperado el 21 de septiembre de 2026, de https://github.com/SebastianMacaluso/TopTagComparison
- <a id="ref-reinhardt-2024"></a>Reinhardt, E. A. F., Dinesh, P. R., & Gleyzer, S. (2024). SineKAN: Kolmogorov-Arnold networks using sinusoidal activation functions. *Frontiers in Artificial Intelligence, 7*. https://doi.org/10.3389/frai.2024.1462952
- <a id="ref-sidharth-2024"></a>Sidharth, S. S., Gokul, R., Anas, K. P., & Keerthana, A. R. (2024). *Chebyshev polynomial-based Kolmogorov-Arnold networks: An efficient architecture for nonlinear function approximation*. arXiv. https://arxiv.org/abs/2405.07200
- <a id="ref-werner-2025"></a>Werner, Y., Malemath, A., Liu, M., Fortes Rey, V., Palaiodimopoulos, N., Lukowicz, P., & Kiefer-Emmanouilidis, M. (2025). QuKAN: A quantum circuit Born machine approach to quantum Kolmogorov Arnold networks. *Scientific Reports, 15*. https://doi.org/10.1038/s41598-025-22705-9
