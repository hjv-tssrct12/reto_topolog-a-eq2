# Reporte Técnico: Análisis Topológico de Datos en la Asociación entre Ácido Fólico y Características Materno-Neonatales

---

## Índice

1. [Contexto y pregunta de investigación](#1-contexto-y-pregunta-de-investigación)
2. [Evolución metodológica](#2-evolución-metodológica)
   - 2.1 [Etapa 2 — Comprensión exploratoria de los datos](#21-etapa-2--comprensión-exploratoria-de-los-datos)
   - 2.2 [Etapa 3 — Regresiones lineales y análisis de colinealidad](#22-etapa-3--regresiones-lineales-y-análisis-de-colinealidad)
   - 2.3 [Etapa 4a — Isomap: geometría no lineal](#23-etapa-4a--isomap-geometría-no-lineal)
   - 2.4 [Etapa 4b — Mapper TDA inicial](#24-etapa-4b--mapper-tda-inicial)
3. [Pipeline final: arquitectura general](#3-pipeline-final-arquitectura-general)
4. [Descripción detallada del pipeline](#4-descripción-detallada-del-pipeline)
   - 4.1 [Fase 1-2: Carga y preparación de datos](#41-fase-1-2-carga-y-preparación-de-datos)
   - 4.2 [Fase 3: Homología persistente](#42-fase-3-homología-persistente)
   - 4.3 [Fase 4: UMAP con grid search](#43-fase-4-umap-con-grid-search)
   - 4.4 [Fase 5a: Barrido multiescala del Mapper](#44-fase-5a-barrido-multiescala-del-mapper)
   - 4.5 [Fase 5b: Grid search multi-métrica](#45-fase-5b-grid-search-multi-métrica)
   - 4.6 [Fase 6: Análisis de estabilidad](#46-fase-6-análisis-de-estabilidad)
   - 4.7 [Fase 8: Análisis estadístico por nodo](#47-fase-8-análisis-estadístico-por-nodo)
   - 4.8 [Fase 9: Machine Learning](#48-fase-9-machine-learning)
   - 4.9 [Fase 9b: Importancia de variables](#49-fase-9b-importancia-de-variables)
   - 4.10 [Fase 7: Visualizaciones](#410-fase-7-visualizaciones)
   - 4.11 [Fase 10: Exportación](#411-fase-10-exportación)
5. [Decisiones técnicas clave](#5-decisiones-técnicas-clave)
6. [Comparativa entre etapas](#6-comparativa-entre-etapas)
7. [Limitaciones y extensiones posibles](#7-limitaciones-y-extensiones-posibles)

---

## 1. Contexto y pregunta de investigación

El proyecto analiza si el nivel de consumo de ácido fólico (AF) durante el embarazo se asocia con la formación de **subgrupos diferenciables** en características maternas y neonatales. El análisis opera sobre un espacio de nueve variables:

| Grupo | Variable |
|-------|----------|
| **Maternas** | Edad madre, Educación, N° embarazo, Región, Δ peso, Δ IMC |
| **Neonatales** | Sexo hijo, Peso al nacer (g), Edad gestacional (sem) |

La variable de interés es el **consumo total de AF** (suplementos + pan), medido en mg/día y categorizado en tres niveles: *Bajo*, *Medio* y *Alto* (percentiles 0-33, 33-66, 66-100). El dataset comprende **N = 1 419 pacientes** provenientes de dos bases de datos fusionadas.

La hipótesis central es que los datos materno-neonatales no se distribuyen de forma uniforme en el espacio 9D, sino que presentan **estructura topológica** —grupos, ciclos, conectividad no trivial— que se correlaciona con el nivel de ingesta de AF. Para explorar esta hipótesis se recurrió al Análisis Topológico de Datos (TDA), particularmente al algoritmo Mapper.

---

## 2. Evolución metodológica

### 2.1 Etapa 2 — Comprensión exploratoria de los datos

**Objetivo**: Familiarizarse con la distribución y calidad de las variables antes de cualquier modelado.

**Métodos**: Estadísticas descriptivas, visualización de distribuciones, evaluación de missingness y exploración de las dos bases de datos originales (base principal + base consenso con diferentes convenciones de nombres de columna).

**Hallazgo principal**: La base consenso codificaba valores faltantes como la cadena de texto `"x"`, lo que requirió limpieza explícita. Además, las columnas categóricas (*Educación*, *Región*, *Sexo hijo*) estaban almacenadas como texto, lo que exigía un paso de codificación ordinal antes de cualquier análisis cuantitativo.

**Limitación**: Análisis puramente descriptivo, sin capacidad de capturar relaciones multivariadas.

---

### 2.2 Etapa 3 — Regresiones lineales y análisis de colinealidad

**Archivo**: `Codigos/etapa3_regresiones_AF.py`

**Objetivo**: Identificar asociaciones simples entre el consumo de AF y cada variable materno-neonatal mediante regresión OLS univariada.

**Metodología**:

1. **Fusión de bases**: Se concatenaron la base principal y la base consenso. Las "x" se reemplazaron por 0; las columnas con nombres distintos se renombraron al esquema común. El DFE (*Dietary Folate Equivalents*) no disponible en la base consenso se imputó como `NaN`.

2. **Codificación ordinal**: Se aplicó `encode_if_object()` a *Educación*, *Región* y *Sexo hijo*. Cualquier columna restante de tipo `object` se forzó a numérico con `pd.to_numeric(..., errors="coerce")`.

3. **Categorización AF**: Se calcularon los percentiles 33 y 66 de cada variable AF y se crearon etiquetas *Bajo / Medio / Alto* con `pd.cut`. Esta base categorizada se guardó como `BD/DB_Limpia_TDA.xlsx`, que sirve de entrada a todas las etapas posteriores.

4. **Modelos**: 4 predictores (Total AF, AF Suplementos 1T, AF Pan, Total DFE) × 9 outcomes = **36 regresiones OLS simples** (`statsmodels.OLS`). Se reportaron β, SE, t, p-valor, IC 95%, R² y R² ajustado.

5. **Colinealidad**: Matriz de correlación de Pearson entre los cuatro predictores AF + **VIF** (*Variance Inflation Factor*). Un VIF > 10 indica colinealidad severa, relevante para decidir si incluir todos los predictores simultáneamente.

**Outputs generados**:
- `BD/DB_Limpia_TDA.xlsx` — base limpia con categorías AF
- `Resultados/Regresiones_AF_DFE.xlsx` — regresiones, VIF, correlaciones, estadísticas descriptivas
- `Resultados/correlacion_AF_DFE.png` — heatmap de correlaciones AF/DFE
- `Resultados/reg_TotalAF_EdadMadre.png` — ejemplo de gráfico de dispersión + línea de regresión

**Limitaciones identificadas**:
- Análisis puramente univariado: cada regresión ignora el efecto de las demás variables.
- OLS asume linealidad y normalidad de residuos; no es adecuado para outcomes categóricos (Sexo, Región).
- No captura la estructura multidimensional del espacio de características.

---

### 2.3 Etapa 4a — Isomap: geometría no lineal

**Archivo**: `Codigos/etapa4_isomap_TDA.py`

**Objetivo**: Descubrir la geometría intrínseca del espacio 9D preservando relaciones geodésicas (distancias a lo largo de la variedad de datos), a diferencia de PCA que solo preserva varianza lineal.

**Metodología**:

1. **Preprocesamiento**: `StandardScaler` sobre las 9 variables feature (media 0, std 1).

2. **Isomap** (`sklearn.manifold.Isomap`, `n_neighbors=10`, `n_components=2`): Construye un grafo de k-vecinos, calcula distancias geodésicas entre todos los pares como camino más corto en el grafo, y aplica MDS (Multidimensional Scaling) sobre esa matriz de distancias para obtener el embedding 2D.

3. **Visualización**: Dos scatter plots del embedding 2D: uno coloreado por AF continuo (escala `viridis`) y otro por categoría AF (paleta discreta Bajo/Medio/Alto).

4. **Correlaciones embedding-features**: Coeficiente de Pearson entre cada coordenada Isomap y cada variable original. Permite interpretar qué dimensiones del espacio de datos captura cada eje del embedding.

5. **Clustering sobre embedding**: `DBSCAN(eps=0.55, min_samples=15)` sobre el espacio 2D para detectar agrupaciones densas. Se perfiló cada cluster por AF promedio y por composición de categorías.

**Hallazgos**: El embedding reveló que los datos no son linealmente separables. Las dimensiones Isomap se correlacionaron de forma diferencial con variables maternas vs. neonatales, sugiriendo que el espacio tiene una estructura geométrica no trivial asociada al perfil de AF.

**Limitación principal**: Isomap es un método de visualización; no construye explícitamente la *topología* del espacio (no detecta ciclos, no genera grafos de nodos con composición). El paso de DBSCAN sobre el embedding 2D es también heurístico y sensible a los parámetros `eps` y `min_samples`, que se fijaron ad hoc.

---

### 2.4 Etapa 4b — Mapper TDA inicial

**Archivo**: `Codigos/etapa4_mapper_TDA.py`

**Objetivo**: Aplicar el algoritmo Mapper para construir un grafo simplicial que represente la topología del espacio de features bajo la lente del consumo de AF.

**El algoritmo Mapper opera en tres pasos**:
1. Proyectar los datos mediante una función de lente (aquí: UMAP 2D).
2. Cubrir el espacio de la lente con una colección de intervalos solapados (*cover*).
3. Dentro de cada celda del cover, clusterizar los puntos; cada cluster se convierte en un nodo del grafo; dos nodos se conectan si comparten al menos un punto.

**Decisiones de implementación**:

| Componente | Elección | Justificación |
|------------|----------|---------------|
| Lente | UMAP 2D (`n_neighbors=15`, `min_dist=0.1`, `seed=42`) | Preserva estructura local mejor que PCA o t-SNE para datasets de mediano tamaño |
| Escalado | MinMaxScaler → [0, 1] | Requerido para que KeplerMapper interprete correctamente el cover |
| Cover | `n_cubes=8`, `overlap=0.50` | Valores ad hoc; 8 cubos por dimensión crea hasta 64 celdas; 50% de solapamiento garantiza conexiones entre nodos adyacentes |
| Clusterer | `DBSCAN(eps=0.42, min_samples=4)` | Parámetros fijos; epsilon elegido visualmente |
| Herramienta | `kmapper 2.1.0` (KeplerMapper) | Librería estándar para Mapper en Python; genera HTML interactivo |

**Post-procesamiento**: Para cada nodo del grafo se calculó la composición de categorías AF (N_Bajo, N_Medio, N_Alto, %) y el AF medio, generando un gráfico de barras apiladas y un archivo Excel con estadísticas por nodo.

**Limitaciones identificadas** (que motivan el pipeline moderno):
- **Parámetros fijos sin justificación cuantitativa**: n_cubes, overlap y los parámetros de DBSCAN se eligieron ad hoc.
- **Sin validación de estabilidad**: No hay evidencia de que el grafo sea robusto ante perturbaciones.
- **Sin inferencia estadística por nodo**: Se calculan porcentajes pero no p-valores ni tamaños de efecto.
- **Sin evaluación de la calidad de la lente UMAP**: Los parámetros de UMAP tampoco se optimizan.
- **Código monolítico**: Todo en un script; sin modularidad, sin configuración centralizada, sin logging.

---

## 3. Pipeline final: arquitectura general

El pipeline moderno (`pipeline_mapper/`) responde directamente a cada limitación identificada en las etapas anteriores. Está organizado en **10 fases** orquestadas por `main.py`, con módulos independientes y configuración centralizada en `config.py`.

```
Phase 1-2  data_loading.py         Carga, limpieza, encoding, escalado dual
     ↓
Phase 3    topological_exploration.py   Homología persistente (Ripser, H₀ y H₁)
     ↓
Phase 4    dim_reduction.py         UMAP grid search (18 configs, 3 métricas)
     ↓
Phase 5a   mapper_multiscale.py     Barrido multiescala (272 covers × 3 clusterers)
     ↓
Phase 5b   mapper_grid_search.py    Grid search multi-métrica (score compuesto)
     ↓
           → Selección config final (grid search winner)
     ↓
Phase 6    stability.py             Bootstrap (50 iter) + Perturbación (20 iter)
     ↓
Phase 8    analysis.py              Análisis estadístico por nodo (FDR + Cliff δ)
     ↓
Phase 9    ml_grid_search.py        Grid search ML: 8 regresores + 6 clasificadores
     ↓
Phase 9b   feature_analysis.py      Importancia de variables (3 métodos compuestos)
     ↓
Phase 7    visualization.py         HTMLs interactivos + PNGs estáticos
     ↓
Phase 10   export.py                Excel con 12 hojas + pip versions
```

**Configuración centralizada** (`src/config.py`): Todos los hiperparámetros —rutas, nombres de columna, grids de búsqueda, semillas, umbrales estadísticos— se definen en un único archivo. Esto asegura reproducibilidad total (SEED = 42) y facilita la experimentación sin tocar el código de los módulos.

---

## 4. Descripción detallada del pipeline

### 4.1 Fase 1-2: Carga y preparación de datos

**Módulo**: `src/data_loading.py`

**`load_and_clean()`**:
1. Lee `BD/DB_Limpia_TDA.xlsx` (producto de la Etapa 3).
2. Valida que todas las columnas requeridas estén presentes mediante `assert`.
3. Elimina filas con `NaN` en la variable lente (AF total).
4. Aplica `_encode_categoricals()`: convierte columnas `object` de `CATEGORICAL_ENCODE_COLS` a enteros ordinales; fuerza a numérico las columnas de `NUMERIC_FORCE_COLS`; un catch-all convierte cualquier columna restante de tipo `object` en `FEATURE_COLS`.
5. Imputa con la mediana columna a columna para cualquier `NaN` restante en features.
6. Verifica con `assert` que no quedan `NaN`.
7. Extrae `lente_vals` (float), `cat_vals` (str: "Bajo"/"Medio"/"Alto"), y `cat_enc` (0/1/2).

**`prepare_features()`**: Aplica dos escaladores sobre la misma matriz raw:
- **MinMaxScaler** → X ∈ [0, 1]: normalización rango, preserva distribución original.
- **RobustScaler → MinMaxScaler** → X ∈ [0, 1]: el RobustScaler usa mediana e IQR (robusto a outliers), luego se re-normaliza a [0, 1] para compatibilidad con UMAP. Ambas versiones se propagan a la Fase 4 para selección por grid search.

**Decisión técnica**: El escalado dual permite que la Fase 4 determine empíricamente cuál escalador produce mejores embeddings UMAP, en lugar de asumir uno a priori.

---

### 4.2 Fase 3: Homología persistente

**Módulo**: `src/topological_exploration.py`

**Objetivo**: Antes de construir el Mapper, caracterizar la topología intrínseca del espacio 9D mediante *homología persistente*, que detecta características topológicas (componentes conexas, ciclos, cavidades) a múltiples escalas de resolución.

**`compute_persistence()`**: Aplica **Vietoris-Rips** sobre la matriz X (escala MinMax) usando `ripser`. Parámetros:
- `maxdim=1`: calcula H₀ (componentes) y H₁ (ciclos/loops de 1D).
- `thresh=0.8`: limita la filtración para controlar tiempo de cómputo con N = 1 419.

El resultado son diagramas de persistencia: pares (nacimiento, muerte) de cada característica topológica. Una característica con gran diferencia (muerte − nacimiento) es *persistente* y, por tanto, genuina (no artefacto).

**`analyze_h1_features()`**: Filtra los ciclos H₁ finitos, calcula su persistencia, y determina cuántos superan el percentil 90 (umbral configurable). Si existe al menos un ciclo persistente, el pipeline lo registra como evidencia de *estructura circular* en el espacio de datos, lo que sugiere que el grafo Mapper podría presentar loops.

**Decisión técnica**: Este análisis no modifica ningún parámetro posterior directamente, pero contextualiza los resultados del Mapper. Un espacio con ciclos H₁ persistentes corresponde conceptualmente a un Mapper que debería exhibir ciclos (β₁ > 0), y este comportamiento se verifica en la Fase 5b como parte del score.

---

### 4.3 Fase 4: UMAP con grid search

**Módulo**: `src/dim_reduction.py`

**Objetivo**: Seleccionar la mejor configuración de UMAP para la lente 2D del Mapper, en lugar de fijar parámetros ad hoc como en la Etapa 4b.

**Grid de búsqueda**: 2 escaladores × 3 `n_neighbors` × 3 `min_dist` = **18 configuraciones**.

Los valores de `n_neighbors` se calibraron en función del tamaño muestral: `[sqrt(N)//2, sqrt(N), 2*sqrt(N)]` = `[18, 37, 74]` para N = 1 419. Esto sigue la heurística de que `n_neighbors` debería escalar con la raíz del tamaño muestral.

**Métricas de evaluación**:

| Métrica | Definición | Propósito |
|---------|-----------|-----------|
| **Trustworthiness** | Fracción de k-vecinos reales en 9D que permanecen como vecinos en 2D | Mide si el embedding no introduce vecinos "falsos" |
| **Continuity** | Trustworthiness con roles invertidos | Mide si el embedding no pierde vecinos reales |
| **Spearman ρ** | Correlación de rango entre distancias euclídeas en 9D y 2D (sobre todos los pares) | Preservación global de distancias |

**Score de selección**: `(trustworthiness + continuity) / 2`. Se prioriza la preservación local (los dos primeros términos), que es lo más relevante para el Mapper, que opera sobre celdas locales del cover.

El embedding ganador se normaliza a [0, 1] con MinMaxScaler antes de pasarlo a las Fases 5a y 5b.

---

### 4.4 Fase 5a: Barrido multiescala del Mapper

**Módulo**: `src/mapper_multiscale.py`

**Objetivo**: Explorar sistemáticamente el espacio de parámetros del Mapper y entender cómo varía la topología del grafo resultante.

#### Clusterers implementados

El pipeline implementa tres clusterers con interfaz compatible con kmapper (`fit_predict(X)`):

**`AdaptiveDBSCAN`** (extiende DBSCAN clásico):
- Ajusta `eps` dinámicamente por celda: calcula las distancias al k-ésimo vecino de cada punto dentro de la celda y usa el percentil 10 de esas distancias como `eps`.
- Parámetros: `k=5`, `percentile=10`, `min_samples=3`.
- Ventaja: No requiere un epsilon global; se adapta a la densidad local de cada celda del cover. En celdas densas el eps es pequeño (clusters finos); en celdas dispersas es mayor (clusters gruesos).

**`HDBSCANClusterer`** (Hierarchical DBSCAN):
- Usa `sklearn.cluster.HDBSCAN` (`min_cluster_size=3`, `min_samples=1`).
- HDBSCAN construye una jerarquía de clusters y extrae los más estables automáticamente, sin necesidad de especificar `eps`.
- Manejo de celdas pequeñas: si `len(X) ≤ min_cluster_size`, todos los puntos se asignan a un único cluster (cluster 0). Si todos los puntos son ruido (etiqueta -1), también se unifican.
- Ventaja respecto a DBSCAN: más robusto ante densidades heterogéneas dentro de una celda.

**`AdaptiveAgglomerative`** (Agglomerative clustering con threshold adaptativo):
- Usa `sklearn.cluster.AgglomerativeClustering` con `linkage='single'` y `distance_threshold` adaptativo.
- El threshold se calcula igual que en `AdaptiveDBSCAN`: percentil k-NN. Esto hace que ambos métodos operen en la misma escala de densidad, haciendo comparables sus resultados.
- Single linkage tiende a producir clusters elongados siguiendo la forma de la variedad de datos, lo cual es apropiado para Mapper donde las celdas pueden tener geometrías no esféricas.
- Ventaja: Captura estructuras que DBSCAN podría fragmentar cuando los puntos forman cadenas continuas.

**Decisión de diseño**: Los tres clusterers se controlan desde la lista `_CLUSTERERS` en `mapper_multiscale.py`. Agregar o quitar un clusterer es una operación de una línea, sin tocar el resto del pipeline.

#### Espacio de búsqueda del cover

| Parámetro | Valores | Criterio de selección |
|-----------|---------|----------------------|
| `n_cubes` | [5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 20, 25, 30, 35, 40, 50] (17 valores) | Rango desde cover grueso (5, pocos nodos) hasta fino (50, muchos nodos pequeños) |
| `overlap` | [0.10, 0.15, ..., 0.85] (16 valores) | Rango desde mínimo solapamiento (10%, pocas aristas) hasta alto (85%, grafo muy conectado) |

**Total**: 17 × 16 × 3 clusterers = **816 configuraciones**.

#### Métricas topológicas por configuración

Para cada uno de los 816 grafos se calculan: n_nodes, n_edges, n_components y n_cycles = max(0, E − V + C) (primer número de Betti β₁, ciclos independientes).

**Criterio de selección** en Fase 5a: La configuración cuyo `n_nodes` esté más cerca de la **mediana global** de todos los 816 grafos. Este criterio es deliberadamente simple: su propósito es elegir un representante "de punto medio" del espacio explorado, no la mejor configuración posible. La selección óptima queda para la Fase 5b.

---

### 4.5 Fase 5b: Grid search multi-métrica

**Módulo**: `src/mapper_grid_search.py`

**Objetivo**: Seleccionar la configuración de Mapper óptima mediante un **score compuesto** que equilibra múltiples criterios de calidad —topológicos, estadísticos y de utilidad analítica—, en lugar de optimizar una sola métrica.

#### Score compuesto

Cada configuración genera 5 métricas normalizadas a [0, 1] (min-max dentro del conjunto de 816 configs), que se combinan con pesos explícitos:

| Métrica | Peso | Descripción | Justificación |
|---------|------|-------------|---------------|
| `lcc_fraction` | 0.25 | Fracción de nodos en el componente conectado más grande | Se prefiere un grafo cohesivo (una estructura) sobre uno fragmentado en islas |
| `cat_separation` | 0.25 | Reducción de impureza Gini entre nodos vs. la mezcla global | Objetivo biológico: los nodos deberían separar bien Bajo/Medio/Alto AF |
| `node_cohesion` | 0.20 | Silhouette score promedio intra-nodo | Nodos compactos en el espacio 9D tienen mayor poder interpretativo |
| `overlap_ratio` | 0.15 | Fracción de puntos que pertenecen a más de un nodo | El solapamiento es la esencia del Mapper; muy poco = grafo rígido, sin estructura |
| `n_cycles` | 0.15 | Número de ciclos independientes (β₁) | Los ciclos H₁ detectados en Fase 3 deberían manifestarse aquí |

```
composite_score = 0.25·lcc_n + 0.25·cat_sep_n + 0.20·cohesion_n + 0.15·overlap_n + 0.15·cycles_n
```

#### Penalización sigmoidal por tamaño

Grafos con demasiados nodos son problemáticos para la Fase 8: nodos con pocos pacientes carecen de potencia estadística para el test de Mann-Whitney. Se aplica una penalización suave:

```
penalty(n) = 1 / (1 + exp(0.03 × (n − 250)))
```

- n ≤ 180 nodos → penalty ≥ 0.87 (casi sin descuento)
- n = 250 nodos → penalty = 0.50 (score a la mitad)
- n > 300 nodos → penalty < 0.18 (configuración prácticamente descartada)

```
penalised_score = composite_score × penalty(n_nodes)
```

**Decisión técnica**: La penalización sigmoid es más suave que un corte duro (ej., descartar todo > 250 nodos). Esto permite que configuraciones ligeramente por encima del umbral aún compitan si tienen métricas de calidad excepcionales.

#### Selección final

La configuración ganadora (máximo `penalised_score`) es adoptada como `final_config`. El grafo correspondiente se reconstruye de forma determinista usando la misma clase de clusterer ganadora.

**Nota sobre el resultado dual**: El pipeline retiene también el ganador de la Fase 5a (mediana de nodos). Ambos se exportan al reporte Excel para comparación. El ganador de la Fase 5b es el que se usa para todas las fases downstream.

---

### 4.6 Fase 6: Análisis de estabilidad

**Módulo**: `src/stability.py`

**Objetivo**: Validar que el grafo Mapper seleccionado no es un artefacto de la muestra particular o de pequeñas variaciones en los datos.

#### Fase 6a — Estabilidad bootstrap (50 iteraciones)

**Protocolo**: En cada iteración se extrae una muestra bootstrap (con reemplazo, n = N), se reconstruye el Mapper con `final_config`, y para cada paciente se registra en qué categoría-dominante de nodo aterrizó. Al final se computa la probabilidad empírica de cada paciente de pertenecer a un nodo dominado por cada categoría AF.

**Output**: DataFrame con columnas `prob_alto_af`, `prob_medio_af`, `prob_bajo_af`, `n_appearances` por paciente. Un paciente con `prob_alto_af ≈ 1.0` es consistentemente asignado a nodos de alta ingesta de AF en prácticamente todos los remuestreos.

**Decisión técnica**: El bootstrap re-entrena el Mapper desde cero en cada iteración (en lugar de solo perturbar etiquetas), lo que refleja la variabilidad real de todo el proceso de clustering dentro de las celdas, no solo del mapeo.

#### Fase 6b — Estabilidad por perturbación (20 iteraciones)

**Protocolo**: Se añade ruido gaussiano con σ = 0.01 a la matriz X normalizada (equivale a perturbaciones sub-percentil en los datos escalados), se reconstruye el Mapper, y se mide la similitud entre el grafo original y el perturbado.

**Métrica**: **Jaccard medio** entre nodos. Para cada nodo del grafo original se encuentra el nodo más similar en el grafo perturbado (mayor Jaccard de sus conjuntos de pacientes) y se promedia. Un valor cercano a 1 indica que los nodos del grafo original son casi idénticos en los grafos perturbados.

```
sim(G_orig, G_pert) = (1/|nodes_orig|) × Σ_{v ∈ nodes_orig} max_{u ∈ nodes_pert} J(v, u)
```

**Output**: Media ± std de similitudes a lo largo de las 20 perturbaciones.

---

### 4.7 Fase 8: Análisis estadístico por nodo

**Módulo**: `src/analysis.py`

**Objetivo**: Determinar, para cada nodo del grafo Mapper, si la distribución de AF de sus pacientes difiere significativamente de la del resto de la cohorte, con corrección por comparaciones múltiples y cuantificación del tamaño de efecto.

**Por cada nodo** se computan:

1. **AF medio e IC 95% bootstrap** (1 000 remuestreos): El IC refleja la incertidumbre muestral del estimador puntual dentro del nodo.

2. **Test de Mann-Whitney U** (bilateral): Compara AF del nodo vs. todos los demás pacientes. Se elige Mann-Whitney por ser no paramétrico (no asume normalidad) y apropiado para muestras de tamaño variable.

3. **Corrección FDR Benjamini-Hochberg**: Ajusta los p-valores de todos los nodos simultáneamente, controlando la tasa de descubrimientos falsos. Umbral: p_FDR < 0.05.

4. **Cliff's delta (δ)**: Tamaño de efecto no paramétrico.
   ```
   δ = (#{x_i > y_j} - #{x_i < y_j}) / (n_nodo × n_resto)
   ```
   Toma valores en [-1, 1]. Umbral de relevancia práctica: |δ| > 0.33 (efecto mediano).

**Criterio de significancia compuesto**: Un nodo es declarado *significativo* si y solo si cumple **ambas** condiciones: `p_FDR < 0.05` AND `|δ| > 0.33`. Esto evita que diferencias estadísticamente significativas pero trivialmente pequeñas (o viceversa) sean reportadas como hallazgos relevantes.

**Decisión técnica**: La exigencia del doble criterio (significancia estadística + relevancia práctica) es una convención creciente en ciencias de la salud. El p-valor solo no distingue si una diferencia es clínica o biológicamente relevante.

---

### 4.8 Fase 9: Machine Learning

**Módulo**: `src/ml_grid_search.py`

**Objetivo**: Evaluar en qué medida las 9 variables materno-neonatales permiten predecir el nivel de AF, tanto en su forma continua (regresión) como categórica (clasificación).

#### Regresión (AF continuo → mg/día)

**Modelos evaluados**: LinearRegression, Ridge, Lasso, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, SVR, KNeighborsRegressor.

**Grid search**: `GridSearchCV` con 5-fold CV, métrica R². Los hiperparámetros de cada modelo se detallan en el código; para modelos sin hiperparámetros (LinearRegression) se usa `cross_val_score` directamente.

#### Clasificación (AF categórico → Bajo/Medio/Alto)

**Modelos evaluados**: LogisticRegression, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, SVC, KNeighborsClassifier.

**Decisiones técnicas clave**:

- **Métrica principal: f1_macro** en lugar de accuracy. Con tres clases aproximadamente balanceadas (≈33% cada una), accuracy y f1_macro son similares numéricamente, pero f1_macro reporta con más transparencia cómo funciona el modelo en cada clase por separado. Si una clase fuera minoritaria, accuracy podría enmascarar un clasificador que simplemente ignora esa clase.

- **StratifiedKFold**: Garantiza que cada fold preserve las proporciones de clase. Esto es importante cuando las categorías son resultado de un corte en percentiles (distribución aproximadamente uniforme) pero podría no serlo en un conjunto desbalanceado.

- **Reporte por clase**: Para el mejor modelo encontrado se acumulan predicciones de todos los folds (validación cruzada manual sobre el mejor estimador) y se genera un `classification_report` con precisión, recall, F1 y soporte para cada clase Bajo/Medio/Alto.

---

### 4.9 Fase 9b: Importancia de variables

**Módulo**: `src/feature_analysis.py`

**Objetivo**: Identificar qué variables materno-neonatales son más relevantes para distinguir los niveles de AF, usando tres métodos complementarios y combinando sus rankings.

#### Método 1: Kruskal-Wallis + FDR

Para cada variable feature, se aplica el test de Kruskal-Wallis (no paramétrico, compara k grupos) entre los tres niveles AF. El tamaño de efecto se cuantifica con **eta-cuadrado**:

```
η² = (H − k + 1) / (n − k)
```

donde H es el estadístico Kruskal-Wallis, k = 3 grupos, n el tamaño total. Interpretación estándar: η² ≥ 0.01 (pequeño), ≥ 0.06 (mediano), ≥ 0.14 (grande). Los p-valores se corrigen con FDR Benjamini-Hochberg.

#### Método 2: Importancias Random Forest (impurity-based)

Se entrena un `RandomForestRegressor(n_estimators=300)` sobre AF continuo y un `RandomForestClassifier(n_estimators=300)` sobre AF categórico. Las importancias impurity-based (media de decrementos en impureza de Gini por feature, ponderada por fracción de muestras que pasan por cada split) se promedian entre ambos modelos.

**Limitación conocida**: Las importancias impurity-based tienen sesgo hacia features de alta cardinalidad. Por eso se complementan con el Método 3.

#### Método 3: Importancias por permutación (model-agnostic)

Para un fold de entrenamiento/validación del mismo RF (regresor y clasificador), se mide el drop en R² / f1_macro cuando cada feature es permutada aleatoriamente (10 repeticiones). Una feature cuya permutación reduce fuertemente el score es genuinamente importante.

**Ventaja**: Model-agnostic; no está influenciado por la estructura del árbol ni la cardinalidad de la variable.

#### Ranking compuesto

Para cada feature se obtienen tres rankings independientes (uno por método). El **composite_rank** es el promedio de los tres rankings (1 = más importante). Esta agregación de rankings es robusta ante cualquier método individual que pudiera estar sesgado.

---

### 4.10 Fase 7: Visualizaciones

**Módulo**: `src/visualization.py`

Se generan seis visualizaciones sobre el grafo Mapper final:

| Archivo | Tipo | Variable de color |
|---------|------|-------------------|
| `mapper_AF.html` | KeplerMapper HTML interactivo | AF total normalizado (continuo) |
| `mapper_cat_AF.html` | KeplerMapper HTML interactivo | Categoría AF codificada (0=Bajo, 1=Medio, 2=Alto) |
| `mapper_densidad.html` | KeplerMapper HTML interactivo | Densidad (n pacientes por nodo, normalizado) |
| `mapper_AF.png` | NetworkX spring layout estático | AF medio por nodo |
| `mapper_cat_AF.png` | NetworkX spring layout estático | Categoría media por nodo |
| `mapper_composicion.png` | Barras apiladas (100%) | Composición Bajo/Medio/Alto por nodo |

En el gráfico de composición, los nodos estadísticamente significativos (Fase 8) se marcan con ★ sobre la barra. Las barras se ordenan de menor a mayor AF medio, facilitando la lectura de la gradiente de consumo a lo largo del grafo.

---

### 4.11 Fase 10: Exportación

**Módulo**: `src/export.py`

El reporte final es un Excel con **12 hojas**:

| Hoja | Contenido |
|------|-----------|
| `nodos_estadisticas` | Estadísticas completas por nodo (Fase 8) |
| `barrido_multiescala` | Topología de las 816 configuraciones (Fase 5a) |
| `estabilidad_bootstrap` | Probabilidades bootstrap por paciente (Fase 6a) |
| `estabilidad_perturbacion` | Similitudes Jaccard de perturbación (Fase 6b) |
| `homologia_persistente` | Resumen H₀/H₁ (Fase 3) |
| `hiperparametros` | Todos los hiperparámetros + versiones de librerías (`pip freeze`) |
| `interpretacion_clinica` | Resumen ejecutivo: medias de features por grupo AF |
| `ml_regresion` | Ranking de modelos de regresión |
| `ml_clasificacion` | Ranking de clasificadores con métricas por clase |
| `mapper_grid_search` | Las 816 configs ordenadas por penalised_score |
| `mapper_gs_config` | Config ganadora del grid search + explicación de pesos |
| `feature_importance` | Ranking compuesto de importancia de variables |

---

## 5. Decisiones técnicas clave

### 5.1 Tres clusterers adaptativos en lugar de DBSCAN fijo

La decisión más relevante de la Fase 5 es reemplazar un clusterer único de parámetros fijos por tres clusterers con threshold adaptativo a la densidad local de cada celda. El problema con DBSCAN de epsilon fijo es que las celdas del cover pueden tener densidades muy distintas: celdas en zonas densas del embedding UMAP tendrán muchos puntos cercanos, mientras que celdas en zonas dispersas tendrán pocos. Un epsilon global óptimo para las primeras sería demasiado pequeño para las segundas (y viceversa).

La solución adoptada es que cada clusterer calcule su propio umbral de densidad **dentro de cada celda**, basándose en las distancias al k-ésimo vecino local. Esto hace que el resultado sea independiente de la escala global de los datos.

### 5.2 Score compuesto vs. optimización de una sola métrica

Optimizar únicamente el número de nodos (mediana, Fase 5a) o únicamente los ciclos topológicos ignoraría dimensiones importantes de calidad. Por ejemplo, un grafo con muchos ciclos pero nodos heterogéneos en composición AF no es útil para la pregunta de investigación. El score compuesto integra criterios topológicos (LCC, ciclos), estadísticos (cohesión) y biológicos (separación de categorías AF).

### 5.3 Penalización sigmoidal por tamaño de grafo

La penalización resuelve un conflicto entre dos objetivos: más n_cubes produce grafos más detallados (mayor separación) pero también nodos más pequeños (menor potencia estadística en Fase 8). En lugar de un corte duro, la sigmoid degrada suavemente el score a medida que n_nodes supera 250, manteniendo competitividad a configuraciones con, por ejemplo, 260-270 nodos si su score compuesto es excepcional.

### 5.4 Double criterion para significancia de nodos (FDR + Cliff δ)

Un nodo con N = 5 pacientes puede obtener un p-valor Mann-Whitney de 0.04 con una diferencia de AF de 0.1 mg/día: estadísticamente significativo pero biológicamente irrelevante. La exigencia de |δ| > 0.33 actúa como filtro de relevancia práctica complementario al filtro estadístico.

### 5.5 f1_macro sobre accuracy en clasificación

Dado que las categorías AF se definen por percentiles, los tres grupos son aproximadamente iguales en tamaño. Sin embargo, la distinción *Bajo* vs. *Alto* puede ser más fácil de aprender que *Bajo* vs. *Medio* (que están contiguos en la escala continua). f1_macro promedia el F1 de cada clase sin ponderar por frecuencia, por lo que un clasificador que aprende bien *Alto* pero falla en *Medio* tendrá un f1_macro más bajo que accuracy, lo que es más informativo.

### 5.6 Ranking compuesto de importancia de variables

Ningún método individual de importancia de variables es óptimo en todos los escenarios. El Kruskal-Wallis es no paramétrico pero ignora interacciones; las importancias impurity-based de RF son eficientes pero con sesgo de cardinalidad; las importancias por permutación son precisas pero dependen del modelo. Promediando los rankings se obtiene una estimación más robusta que cualquiera por separado.

---

## 6. Comparativa entre etapas

| Dimensión | Etapa 3 | Etapa 4a (Isomap) | Etapa 4b (Mapper) | Pipeline final |
|-----------|---------|-------------------|-------------------|----------------|
| **Modelado** | OLS univariado | Embedding no lineal | Mapper TDA | Mapper TDA + homología |
| **Parámetros** | Implícitos en modelo | Fijos (n_neighbors=10) | Fijos (n_cubes=8, overlap=0.50, eps=0.42) | Grid search sistemático (816 configs) |
| **Selección de config** | N/A | N/A | Ad hoc | Score compuesto multi-métrica |
| **Estabilidad** | No evaluada | No evaluada | No evaluada | Bootstrap (50 iter) + Perturbación (20 iter) |
| **Inferencia por nodo** | No aplica | No aplica | Descriptiva (%) | FDR + Cliff's delta |
| **Multiple testing** | No | No | No | Benjamini-Hochberg |
| **ML predictivo** | No | No | No | 8 regresores + 6 clasificadores |
| **Importancia features** | Implícita en VIF | Correlación lineal | No | 3 métodos + ranking compuesto |
| **Reproducibilidad** | Manual, sin seed | Sin seed explícito | SEED=42 pero monolítico | Config centralizada, SEED=42 en todo |
| **Modularidad** | Script único | Script único | Script único | 10 módulos independientes |
| **Logging** | `print()` | `print()` | `print()` | `logging` estructurado |
| **Outputs** | Excel + PNGs | Excel + PNGs + HTMLs | Excel + HTMLs | Excel 12 hojas + 6 visualizaciones + CSVs intermedios |

---

## 7. Limitaciones y extensiones posibles

### Limitaciones actuales

1. **Análisis observacional**: No es posible establecer causalidad entre niveles de AF y características materno-neonatales.

2. **Categorización percentil-based**: El corte en p33/p66 es arbitrario. Una alternativa sería utilizar umbrales clínicos de ingesta recomendada de ácido fólico.

3. **Espacio de lente fijo**: La lente del Mapper es siempre UMAP 2D sobre el espacio de features. Otras opciones —excentridicidad, distancia a un centroide, proyección sobre la variable AF directamente— podrían revelar estructuras distintas.

4. **Clusterer único por config**: En la Fase 5b, cada configuración usa solo un clusterer. Una extensión sería evaluar ensambles de clusterers (votar cuál es la partición más estable).

5. **Ejecución secuencial**: Con 816 configs × tiempos de clustering + silhouette, la Fase 5b puede tomar varios minutos. No hay paralelización implementada.

### Extensiones posibles

| Extensión | Descripción |
|-----------|-------------|
| Paralelización | `joblib.Parallel` sobre el loop de configs en Fase 5b |
| Más clusterers | Ward linkage, MeanShift, Birch, espectral |
| Otras lentes | Excentricidad topológica, distancia a medoide por clase AF, PCA1 |
| Análisis longitudinal | Si hay datos de seguimiento, Mapper dinámico sobre series temporales |
| Persistence images en ML | Usar el vector de persistence image (Fase 3) como feature adicional en la Fase 9 |
| Validación externa | Aplicar el pipeline a una cohorte independiente con el config ganador fijo |
| Interpretabilidad de nodos | Usar SHAP values por nodo para entender qué variables explican la composición AF |
