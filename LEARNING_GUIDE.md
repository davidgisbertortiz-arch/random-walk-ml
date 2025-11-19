# 📚 Guía de Aprendizaje: Machine Learning en Random Walks

> **Una guía didáctica completa para entender cada concepto, modelo y técnica utilizada en este proyecto**

---

## 🎯 Índice

1. [Conceptos Fundamentales](#-conceptos-fundamentales)
2. [Random Walks Explicados](#-random-walks-explicados)
3. [Feature Engineering](#-feature-engineering)
4. [Modelos de Machine Learning](#-modelos-de-machine-learning)
5. [Validación y Métricas](#-validación-y-métricas)
6. [Group-Aware Validation](#-group-aware-validation-el-concepto-crítico)
7. [Ejercicios Prácticos](#-ejercicios-prácticos)
8. [Recursos Adicionales](#-recursos-adicionales)

---

## 🧠 Conceptos Fundamentales

### ¿Qué es un Random Walk?

Un **random walk** (caminata aleatoria) es un proceso estocástico donde cada paso es aleatorio e independiente de los anteriores.

**Analogía:** Imagina lanzar una moneda:
- ✅ Cara → Das un paso hacia adelante (+1)
- ❌ Cruz → Das un paso hacia atrás (-1)

```python
import numpy as np

# Random walk justo (p=0.5)
steps = np.random.choice([-1, 1], size=100)
position = np.cumsum(steps)  # Posición acumulada

# Tu trayectoria: [0, -1, 0, 1, 0, -1, -2, -1, ...]
```

**Visualización:**

```
Posición
   |
 5 |           *
   |        *     *
 0 |  *  *           *
   | *                  *
-5 |________________________
     0  10  20  30  40  50
              Tiempo
```

### Random Walk "Justo" vs "Sesgado"

| Tipo | Probabilidad | Comportamiento | Ejemplo Real |
|------|--------------|----------------|--------------|
| **Justo** | p(+1) = 0.5 | Sin tendencia, totalmente aleatorio | Mercado eficiente |
| **Sesgado positivo** | p(+1) = 0.7 | Tendencia alcista | Mercado alcista |
| **Sesgado negativo** | p(+1) = 0.3 | Tendencia bajista | Mercado bajista |

**Pregunta clave del proyecto:** ¿Puede ML detectar si un walk es justo o sesgado observando solo una ventana corta de pasos?

---

## 🎲 Random Walks Explicados

### 1. Random Walk 1D (Una Dimensión)

El más simple: movimiento en una línea.

```python
from enhanced_model import WalkConfig, generate_random_walks_1d

# Configuración
config = WalkConfig(
    n_walks=10,        # Generar 10 caminatas
    n_steps=200,       # Cada una con 200 pasos
    bias_mode="mixed"  # Mezcla de justos y sesgados
)

positions, p_ups = generate_random_walks_1d(config)

print(f"Forma de positions: {positions.shape}")  # (10, 200)
print(f"Probabilidades: {p_ups}")  # [0.5, 0.7, 0.3, ...]
```

**¿Qué obtenemos?**
- `positions[i, t]`: Posición del walk `i` en el tiempo `t`
- `p_ups[i]`: Probabilidad de paso +1 para el walk `i`

**Visualización de un walk sesgado:**

```python
import matplotlib.pyplot as plt

# Walk con sesgo positivo (p=0.7)
steps = np.random.choice([-1, 1], size=500, p=[0.3, 0.7])
position = np.cumsum(steps)

plt.plot(position)
plt.axhline(0, color='red', linestyle='--', label='Inicio')
plt.title('Random Walk Sesgado (p=0.7)')
plt.xlabel('Pasos')
plt.ylabel('Posición')
plt.legend()
plt.grid(True, alpha=0.3)
```

**Observa:** El walk tiende a subir (más +1 que -1).

### 2. Random Walk 2D (Dos Dimensiones)

Movimiento en un plano (X, Y).

```python
from enhanced_model import generate_random_walks_nd

# Configuración 2D
config_2d = WalkConfig(
    n_walks=5,
    n_steps=300,
    dimensions=2,  # ¡2D!
    bias_mode="mixed"
)

positions_2d, p_ups_2d = generate_random_walks_nd(config_2d)

print(f"Forma: {positions_2d.shape}")  # (5, 300, 2)
#                                         ↑   ↑    ↑
#                                      walks pasos dims

# Cada walk tiene:
# - positions_2d[i, :, 0] → trayectoria en X
# - positions_2d[i, :, 1] → trayectoria en Y
# - p_ups_2d[i, 0] → sesgo en X
# - p_ups_2d[i, 1] → sesgo en Y
```

**Aplicación práctica:** Robots, drones, partículas en física.

### 3. Mezcla de Walks (Clave del Proyecto)

```python
from enhanced_model import BiasDistribution

# Configurar mezcla
bias_dist = BiasDistribution(
    fair_prob=0.2,              # 20% serán justos (p=0.5)
    positive_bias_prob=0.4,     # 40% sesgo positivo
    negative_bias_prob=0.4,     # 40% sesgo negativo
    positive_bias_range=(0.6, 0.75),  # p entre 0.6 y 0.75
    negative_bias_range=(0.25, 0.4)   # p entre 0.25 y 0.4
)

config = WalkConfig(
    n_walks=100,
    n_steps=500,
    bias_mode="mixed",
    bias_distribution=bias_dist
)
```

**¿Por qué mezclar?**

Si todos fueran justos → ML no aprende nada (no hay señal)  
Si todos fueran iguales → ML aprende pero no generaliza  
**Mezcla realista** → ML aprende patrones útiles que funcionan en el mundo real

---

## 🔧 Feature Engineering

### ¿Qué son Features (Características)?

Features son las **variables de entrada** que el modelo ML usa para hacer predicciones.

**Problema:** Tenemos una secuencia de posiciones `[10, 11, 9, 10, 12, ...]`  
**Objetivo:** Extraer información útil para predecir el sesgo

### 1. Raw Deltas (Diferencias Brutas)

La forma más simple: usar los pasos directamente.

```python
# Ventana de 20 pasos
window = [10, 11, 9, 10, 12, 13, 11, 10, 12, 14, 
          15, 13, 14, 16, 17, 15, 16, 18, 19, 17]

# Calcular deltas (diferencias)
deltas = [11-10, 9-11, 10-9, 12-10, ...]
#       = [1, -2, 1, 2, 1, -2, 1, 2, ...]

# Estas deltas son nuestras features
# Si hay más +1 que -1 → sesgo positivo probable
```

**En código:**

```python
from enhanced_model import FeatureConfig, make_windows_from_walks_enhanced

# Solo raw deltas
feature_config = FeatureConfig(use_raw_deltas=True)

X, y, groups = make_windows_from_walks_enhanced(
    positions,
    window=20,
    feature_config=feature_config
)

print(f"Features por muestra: {X.shape[1]}")  # 20 (el tamaño de ventana)
```

**Ventajas:** Simple, directo  
**Desventajas:** Pierde información agregada

### 2. Statistical Features (Características Estadísticas)

Agregar información resumida de la ventana.

```python
# Misma ventana
deltas = [1, -2, 1, 2, 1, -2, 1, 2, 1, 2, ...]

# Estadísticas
mean = np.mean(deltas)      # Media: ¿tiende a subir o bajar?
std = np.std(deltas)        # Volatilidad: ¿qué tan errático?
skew = scipy.stats.skew(deltas)  # Asimetría: ¿más +1 o -1?
kurtosis = scipy.stats.kurtosis(deltas)  # Colas pesadas
range_val = max(deltas) - min(deltas)  # Rango de movimiento
```

**Interpretación:**

| Estadística | Valor | Interpretación |
|-------------|-------|----------------|
| **Mean > 0** | +0.15 | Tendencia alcista |
| **Mean < 0** | -0.20 | Tendencia bajista |
| **Std alta** | 1.5 | Muy volátil |
| **Skew > 0** | +0.8 | Más valores positivos |
| **Skew < 0** | -0.8 | Más valores negativos |

**En código:**

```python
# Raw deltas + estadísticas
feature_config = FeatureConfig(
    use_raw_deltas=True,
    use_statistics=True,
    statistics=["mean", "std", "skew", "range"]
)

X, y, groups = make_windows_from_walks_enhanced(
    positions,
    window=20,
    feature_config=feature_config
)

print(f"Features por muestra: {X.shape[1]}")  
# 20 (deltas) + 4 (estadísticas) = 24
```

**Resultado típico:** +5-10% mejora en ROC-AUC

### 3. Trend Features (Características de Tendencia)

Capturar la dirección del movimiento.

```python
from scipy.stats import linregress

# Misma ventana
deltas = [1, -2, 1, 2, 1, -2, 1, 2, 1, 2, ...]
time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]

# Regresión lineal
slope, intercept, r_value, _, _ = linregress(time, deltas)

# Features de tendencia
trend_slope = slope         # ¿Sube o baja con el tiempo?
trend_correlation = r_value # ¿Qué tan fuerte es la tendencia?
```

**Interpretación:**

| Feature | Valor | Significado |
|---------|-------|-------------|
| **slope > 0** | +0.05 | Acelerando hacia arriba |
| **slope < 0** | -0.05 | Desacelerando |
| **r_value alto** | 0.8 | Tendencia clara |
| **r_value bajo** | 0.1 | Sin tendencia clara |

**En código:**

```python
# Todas las features
feature_config = FeatureConfig(
    use_raw_deltas=True,
    use_statistics=True,
    use_trend=True,
    statistics=["mean", "std", "skew", "range"]
)

X, y, groups = make_windows_from_walks_enhanced(
    positions,
    window=20,
    feature_config=feature_config
)

print(f"Features totales: {X.shape[1]}")
# 20 (deltas) + 4 (stats) + 2 (trend) = 26
```

**Resultado típico:** +3-8% mejora adicional

### 4. Sliding Windows (Ventanas Deslizantes)

**Concepto clave:** De un walk largo, extraemos múltiples muestras.

```
Walk completo (500 pasos):
[10, 11, 9, 10, 12, 13, 11, 10, 12, 14, 15, 13, 14, 16, ...]

Ventana 1 (pasos 0-19):    [10, 11, 9, 10, 12, 13, 11, 10, 12, 14, 15, 13, 14, 16, 17, 15, 16, 18, 19, 17]
Ventana 2 (pasos 1-20):        [11, 9, 10, 12, 13, 11, 10, 12, 14, 15, 13, 14, 16, 17, 15, 16, 18, 19, 17, 20]
Ventana 3 (pasos 2-21):            [9, 10, 12, 13, 11, 10, 12, 14, 15, 13, 14, 16, 17, 15, 16, 18, 19, 17, 20, 18]
...
```

**Ventaja:** De 100 walks de 500 pasos → ¡Miles de muestras de entrenamiento!

**⚠️ PELIGRO:** Ventanas del mismo walk están correlacionadas → necesitamos **Group-Aware Validation**

---

## 🤖 Modelos de Machine Learning

### 1. Dummy Classifiers (Baselines)

**¿Por qué empezar aquí?** Para saber si realmente estamos aprendiendo algo.

#### Dummy Majority

```python
from enhanced_model import build_pipeline

# Siempre predice la clase mayoritaria
model = build_pipeline("dummy_majority")
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print(f"Accuracy: {score:.3f}")  # ~0.50 si clases balanceadas
```

**Estrategia:** Ignora las features, siempre dice "clase mayoritaria"

**Ejemplo:**
- Si 60% son sesgados positivos, siempre predice "positivo"
- Accuracy: 60% pero no aprendió nada útil

#### Dummy Stratified

```python
model = build_pipeline("dummy_stratified")
# Predice aleatoriamente según proporción de clases
```

**Si tus modelos reales no superan estos baselines → No hay señal en los datos**

### 2. Logistic Regression (Regresión Logística)

**Modelo lineal** más simple para clasificación.

#### ¿Cómo funciona?

```python
# Modelo lineal
z = w1*feature1 + w2*feature2 + ... + wn*featuren + b

# Función sigmoide para probabilidad
probability = 1 / (1 + exp(-z))

# Decisión
if probability > 0.5:
    prediction = "sesgado"
else:
    prediction = "justo"
```

**Visualización 2D:**

```
Feature 2
    |     
  1 |    ●●●    Clase 1 (sesgado)
    |   ●●●
  0 |──────────  Límite de decisión (línea recta)
    | ○○○
 -1 | ○○○○     Clase 0 (justo)
    |________________
       -1   0   1
           Feature 1
```

**Ventajas:**
- ✅ Rápido de entrenar
- ✅ Interpretable (puedes ver los pesos)
- ✅ Funciona bien con features lineales

**Desventajas:**
- ❌ No captura relaciones no lineales
- ❌ Asume independencia entre features

**En código:**

```python
from enhanced_model import build_pipeline

# Crear pipeline
model = build_pipeline("logreg")

# Entrenar
model.fit(X_train, y_train)

# Evaluar
from enhanced_model import evaluate
metrics = evaluate(model, X_test, y_test)

print(f"ROC-AUC: {metrics['roc_auc']:.3f}")  # ~0.55-0.65
print(f"Accuracy: {metrics['accuracy']:.3f}")  # ~0.58-0.68
```

**¿Cuándo usar?**
- Baseline rápido
- Datos linealmente separables
- Necesitas interpretabilidad

### 3. Random Forest (Bosque Aleatorio)

**Ensemble de árboles de decisión** que vota.

#### ¿Cómo funciona?

```
Random Forest = Árbol 1 + Árbol 2 + ... + Árbol 100

Cada árbol:
1. Toma muestra aleatoria de datos
2. Toma subconjunto aleatorio de features
3. Construye árbol de decisión
4. Vota en la predicción final
```

**Visualización de un árbol:**

```
                [Todas las muestras]
                        |
                   mean > 0.1?
                   /          \
                 Sí            No
                 /              \
         [Sesgo positivo]    std > 0.5?
                             /        \
                           Sí          No
                          /              \
                  [Volátil]          [Justo]
```

**Ventajas:**
- ✅ Captura relaciones no lineales
- ✅ Robusto a outliers
- ✅ Feature importance automático
- ✅ No requiere normalización

**Desventajas:**
- ❌ Más lento que regresión logística
- ❌ Menos interpretable
- ❌ Puede sobreajustar con muchos árboles

**Hiperparámetros importantes:**

```python
from enhanced_model import build_pipeline

model = build_pipeline("rf")

# Ver hiperparámetros por defecto
print(model.named_steps['clf'].get_params())

# Hiperparámetros clave:
# - n_estimators: Número de árboles (default: 100)
# - max_depth: Profundidad máxima (default: None)
# - min_samples_split: Min muestras para dividir (default: 2)
# - max_features: Features por árbol (default: 'sqrt')
```

**En código con tuning:**

```python
from enhanced_model import tune_with_cv

# Grid de hiperparámetros
param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [5, 10, None],
    'clf__min_samples_split': [2, 5, 10]
}

# Búsqueda con CV
best_model = tune_with_cv(
    "rf",
    X_train, y_train, groups_train,
    param_grid=param_grid,
    n_splits=5
)

print(f"Mejores parámetros: {best_model.best_params_}")
```

**Performance típico:** ROC-AUC 0.65-0.75

### 4. Histogram Gradient Boosting (HGB)

**El más potente** de los modelos en este proyecto.

#### ¿Cómo funciona?

```
Gradient Boosting = Modelo 1 + Modelo 2 + ... + Modelo N

Iteración 1: Entrena modelo inicial → errores grandes
Iteración 2: Entrena modelo para corregir errores del anterior
Iteración 3: Entrena modelo para corregir errores acumulados
...
Iteración N: Predicción final = suma de todos los modelos
```

**Analogía:** Como un equipo donde cada miembro corrige errores del anterior.

**Histogram-based:** Agrupa features en "bins" (histogramas) para ser más rápido.

```
Feature continua:     [0.1, 0.15, 0.18, 0.22, 0.25, 0.3, ...]
                              ↓
Bins (256 valores):   [Bin 1, Bin 1, Bin 2, Bin 2, Bin 3, ...]
```

**Ventajas:**
- ✅ **Mejor performance** en la mayoría de casos
- ✅ Rápido (gracias a histogramas)
- ✅ Maneja missing values
- ✅ Regularización incorporada

**Desventajas:**
- ❌ Más hiperparámetros para tuning
- ❌ Puede sobreajustar si no se regulariza
- ❌ Menos interpretable que Random Forest

**Hiperparámetros clave:**

```python
from enhanced_model import build_pipeline

model = build_pipeline("hgb")

# Hiperparámetros importantes:
# - max_iter: Número de iteraciones (default: 100)
# - max_depth: Profundidad de árboles (default: None)
# - learning_rate: Tasa de aprendizaje (default: 0.1)
# - min_samples_leaf: Min muestras en hoja (regularización)
# - l2_regularization: Regularización L2 (default: 0)
```

**Configuración recomendada:**

```python
# Para evitar overfitting
model = build_pipeline("hgb")
model.set_params(
    clf__max_depth=6,           # Limitar profundidad
    clf__learning_rate=0.1,      # Learning rate moderado
    clf__min_samples_leaf=20,    # Regularización
    clf__l2_regularization=0.1   # Penalización L2
)

model.fit(X_train, y_train)
```

**Performance típico:** ROC-AUC 0.70-0.80

### Comparación de Modelos

| Modelo | Velocidad | Performance | Interpretabilidad | Cuándo Usar |
|--------|-----------|-------------|-------------------|-------------|
| **Dummy** | ⚡⚡⚡ | ❌ | ✅✅✅ | Baseline |
| **LogReg** | ⚡⚡⚡ | 🟨 | ✅✅✅ | Rápido, lineal |
| **Random Forest** | ⚡⚡ | 🟩 | ✅✅ | No lineal, robusto |
| **HGB** | ⚡⚡ | 🟩🟩 | ✅ | Mejor performance |

**Estrategia recomendada:**

1. **Empieza con Dummy** → Baseline
2. **Prueba LogReg** → ¿Es lineal el problema?
3. **Prueba Random Forest** → ¿Mejora con no linealidad?
4. **Afina HGB** → Máximo performance

---

## 📊 Validación y Métricas

### ¿Por qué no solo Accuracy?

**Problema:** Accuracy puede engañar.

```python
# Dataset: 95% clase 0, 5% clase 1
# Modelo dummy que siempre predice 0
# Accuracy: 95% ¡Pero es inútil!
```

### Métricas Explicadas

#### 1. Accuracy (Exactitud)

```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

- **TP** (True Positive): Predijo sesgado Y era sesgado ✅
- **TN** (True Negative): Predijo justo Y era justo ✅
- **FP** (False Positive): Predijo sesgado pero era justo ❌
- **FN** (False Negative): Predijo justo pero era sesgado ❌

**Cuándo usar:** Clases balanceadas (50/50)

#### 2. Precision (Precisión)

```python
precision = TP / (TP + FP)
```

**Pregunta:** De los que predije como "sesgados", ¿cuántos realmente lo eran?

**Ejemplo:**
- Predije 100 como sesgados
- 80 realmente lo eran
- Precision = 80/100 = 0.80

**Cuándo importa:** Cuando los falsos positivos son costosos (ej. alertas de fraude)

#### 3. Recall (Sensibilidad)

```python
recall = TP / (TP + FN)
```

**Pregunta:** De todos los realmente "sesgados", ¿cuántos detecté?

**Ejemplo:**
- Había 90 sesgados reales
- Detecté 80 de ellos
- Recall = 80/90 = 0.89

**Cuándo importa:** Cuando los falsos negativos son costosos (ej. detección de enfermedades)

#### 4. F1-Score

```python
f1 = 2 * (precision * recall) / (precision + recall)
```

**Qué mide:** Balance entre precision y recall

**Cuándo usar:** Cuando quieres un balance y las clases están desbalanceadas

#### 5. ROC-AUC (Área Bajo la Curva ROC)

**La métrica más importante en este proyecto.**

**ROC Curve:** Gráfica de True Positive Rate vs False Positive Rate

```
True Positive Rate (TPR)
    |
1.0 |    ___/‾‾‾‾‾    ← Modelo perfecto
    |   /
0.8 |  /  ← Nuestro modelo
    | /
0.5 |/____________    ← Modelo aleatorio (diagonal)
    |
0.0 |________________
   0.0  0.5  0.8  1.0
   False Positive Rate (FPR)
```

**AUC (Area Under Curve):**

| AUC | Interpretación |
|-----|----------------|
| **0.50** | Aleatorio (moneda al aire) |
| **0.50-0.60** | Señal muy débil |
| **0.60-0.70** | Señal aceptable |
| **0.70-0.80** | Buena señal |
| **0.80-0.90** | Excelente señal |
| **0.90-1.00** | Casi perfecto (cuidado con overfitting) |

**Por qué usamos ROC-AUC:**
- ✅ Insensible a clases desbalanceadas
- ✅ Mide capacidad de discriminación
- ✅ Independiente del threshold
- ✅ Fácil de interpretar

**En código:**

```python
from enhanced_model import evaluate

metrics = evaluate(model, X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")  # ← Métrica principal
```

#### 6. Matthews Correlation Coefficient (MCC)

```python
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Rango:** -1 (peor) a +1 (perfecto), 0 = aleatorio

**Ventaja:** Funciona bien con clases desbalanceadas

#### 7. Cohen's Kappa

**Mide:** Acuerdo entre predicciones y realidad, ajustado por azar

**Interpretación:**
- κ < 0: Peor que azar
- κ = 0-0.20: Leve
- κ = 0.21-0.40: Aceptable
- κ = 0.41-0.60: Moderado
- κ = 0.61-0.80: Sustancial
- κ = 0.81-1.00: Casi perfecto

---

## 🔒 Group-Aware Validation (El Concepto Crítico)

### El Problema del Data Leakage

**Escenario:** Tienes 100 walks, cada uno genera 480 ventanas (sliding windows).

```
Walk 1: [Ventana 1, Ventana 2, Ventana 3, ..., Ventana 480]
Walk 2: [Ventana 1, Ventana 2, Ventana 3, ..., Ventana 480]
...
Walk 100: [...]

Total: 100 × 480 = 48,000 muestras
```

**❌ MAL - Train/Test Split Normal:**

```python
# ESTO ESTÁ MAL
X_train, X_test = train_test_split(X, y, test_size=0.2)

# Problema: Ventanas del mismo walk están en train Y test
# Walk 1 → Ventanas 1-380 en train, 381-480 en test
# Las ventanas están correlacionadas → LEAKAGE!
```

**Resultado:** Performance inflada artificialmente (overfitting)

**Por qué es malo:**

```
Train: Walk1[ventana 100], Walk1[ventana 101], Walk1[ventana 102], ...
Test:  Walk1[ventana 103], Walk1[ventana 104], ...

¡Ventana 103 es casi idéntica a ventana 102!
El modelo "memoriza" el walk, no aprende patrones generales
```

### La Solución: Group-Aware Validation

**✅ BIEN - Split por Walk Completo:**

```python
# ESTO ESTÁ BIEN
from enhanced_model import group_train_test_split

X_train, X_test, y_train, y_test, g_train, g_test = group_train_test_split(
    X, y, groups,  # ← groups identifica qué ventanas vienen del mismo walk
    test_size=0.2,
    seed=42
)

# Resultado: Si Walk 1 está en test, TODAS sus ventanas están en test
# Train: Walk 1-80 (TODAS sus ventanas)
# Test:  Walk 81-100 (TODAS sus ventanas)
```

**Visualización:**

```
❌ MAL:
Train: [W1-vent1, W1-vent2, W2-vent1, W3-vent1, W1-vent3, ...]
Test:  [W1-vent4, W2-vent2, W3-vent2, ...]
       ↑ LEAKAGE! W1 en ambos sets

✅ BIEN:
Train: [W1-todas, W2-todas, W3-todas, ..., W80-todas]
Test:  [W81-todas, W82-todas, ..., W100-todas]
       ↑ Sin leakage, walks completamente separados
```

### Group K-Fold Cross-Validation

**CV normal (MAL):**

```python
# Fold 1: Train en 80%, test en 20% (walks mezclados)
# Fold 2: Train en 80%, test en 20% (walks mezclados)
# ...
```

**Group K-Fold (BIEN):**

```python
from enhanced_model import tune_with_cv

best_model = tune_with_cv(
    "hgb",
    X_train, y_train, groups_train,  # ← groups parameter
    n_splits=5
)

# Fold 1: Train walks 1-80,   Test walks 81-100
# Fold 2: Train walks 1-60,81-100, Test walks 61-80
# ...
# Cada walk SIEMPRE completo en un solo fold
```

### Implementación en Código

```python
# 1. Generar walks
positions, p_ups = generate_random_walks_1d(config)

# 2. Extraer ventanas (MANTENER groups)
X, y, groups = make_windows_from_walks_enhanced(
    positions,
    window=20
)

print(f"Muestras: {X.shape[0]}")  # 48,000
print(f"Walks únicos: {len(np.unique(groups))}")  # 100

# 3. Split group-aware
X_train, X_test, y_train, y_test, g_train, g_test = group_train_test_split(
    X, y, groups, test_size=0.2
)

# Verificar separación
train_walks = set(g_train)
test_walks = set(g_test)
print(f"Intersección: {train_walks & test_walks}")  # set() - ¡Vacío!

# 4. Entrenar normalmente
model = build_pipeline("hgb")
model.fit(X_train, y_train)

# 5. Evaluar (performance realista)
metrics = evaluate(model, X_test, y_test)
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")  # Performance real, sin leakage
```

### Impacto en Performance

**Ejemplo real:**

| Método | ROC-AUC Train | ROC-AUC Test | Interpretación |
|--------|---------------|--------------|----------------|
| ❌ **Sin Group-Aware** | 0.95 | 0.92 | ¡Sospechoso! Muy alto |
| ✅ **Con Group-Aware** | 0.78 | 0.72 | Realista, generaliza |

**Diferencia de 0.92 → 0.72 = 0.20 (20% del performance era overfitting!)**

### Cuándo Aplicar Group-Aware

**Siempre que tengas:**
- Time series con ventanas deslizantes
- Múltiples muestras del mismo sujeto/entidad
- Datos correlacionados de la misma fuente
- Mediciones repetidas

**Ejemplos:**
- ✅ Random walks (este proyecto)
- ✅ Series temporales financieras
- ✅ Sensores IoT
- ✅ Pacientes en estudios médicos
- ✅ Usuarios en sistemas de recomendación

---

## 🎓 Ejercicios Prácticos

### Ejercicio 1: Detectar Fair vs Biased (Básico)

**Objetivo:** Entrenar tu primer modelo para detectar sesgo.

```python
# 1. Imports
from enhanced_model import (
    WalkConfig, generate_random_walks_1d,
    make_windows_from_walks_enhanced,
    group_train_test_split, build_pipeline, evaluate
)

# 2. Generar datos
config = WalkConfig(
    n_walks=50,
    n_steps=300,
    bias_mode="mixed"
)
positions, p_ups = generate_random_walks_1d(config)

# 3. Features (solo raw deltas)
X, y, groups = make_windows_from_walks_enhanced(positions, window=20)

# 4. Split
X_train, X_test, y_train, y_test, g_train, g_test = group_train_test_split(
    X, y, groups, test_size=0.2
)

# 5. Entrenar
model = build_pipeline("logreg")
model.fit(X_train, y_train)

# 6. Evaluar
metrics = evaluate(model, X_test, y_test)
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

# ¿Pregunta?: ¿Supera 0.50 (baseline)?
```

**Meta:** ROC-AUC > 0.55

### Ejercicio 2: Comparar Feature Engineering

**Objetivo:** Ver el impacto de diferentes features.

```python
from enhanced_model import FeatureConfig

configs = {
    "Solo deltas": FeatureConfig(use_raw_deltas=True),
    "Deltas + Stats": FeatureConfig(
        use_raw_deltas=True,
        use_statistics=True
    ),
    "Todas las features": FeatureConfig(
        use_raw_deltas=True,
        use_statistics=True,
        use_trend=True
    )
}

results = {}
for name, feat_config in configs.items():
    X, y, groups = make_windows_from_walks_enhanced(
        positions,
        window=20,
        feature_config=feat_config
    )
    
    X_train, X_test, y_train, y_test, _, _ = group_train_test_split(
        X, y, groups, test_size=0.2
    )
    
    model = build_pipeline("rf")
    model.fit(X_train, y_train)
    
    metrics = evaluate(model, X_test, y_test)
    results[name] = metrics['roc_auc']
    print(f"{name}: {metrics['roc_auc']:.3f}")

# ¿Pregunta?: ¿Qué configuración es mejor?
```

### Ejercicio 3: Comparar Modelos

**Objetivo:** Encontrar el mejor modelo para tu problema.

```python
models = ["logreg", "rf", "hgb"]

for model_name in models:
    model = build_pipeline(model_name)
    model.fit(X_train, y_train)
    
    metrics = evaluate(model, X_test, y_test)
    print(f"\n{model_name.upper()}:")
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1-Score: {metrics['f1']:.3f}")

# ¿Pregunta?: ¿Qué modelo es mejor? ¿Vale la pena la complejidad?
```

### Ejercicio 4: Demostrar Data Leakage

**Objetivo:** Ver por ti mismo el problema del leakage.

```python
from sklearn.model_selection import train_test_split

# MAL: Split normal (con leakage)
X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_bad = build_pipeline("hgb")
model_bad.fit(X_train_bad, y_train_bad)
metrics_bad = evaluate(model_bad, X_test_bad, y_test_bad)

# BIEN: Split group-aware
X_train_good, X_test_good, y_train_good, y_test_good, _, _ = group_train_test_split(
    X, y, groups, test_size=0.2, seed=42
)

model_good = build_pipeline("hgb")
model_good.fit(X_train_good, y_train_good)
metrics_good = evaluate(model_good, X_test_good, y_test_good)

# Comparar
print(f"❌ Sin Group-Aware: ROC-AUC = {metrics_bad['roc_auc']:.3f}")
print(f"✅ Con Group-Aware: ROC-AUC = {metrics_good['roc_auc']:.3f}")
print(f"Diferencia: {(metrics_bad['roc_auc'] - metrics_good['roc_auc']):.3f}")

# ¿Pregunta?: ¿Cuánto overfitting hay sin group-aware?
```

### Ejercicio 5: Análisis de Feature Importance

**Objetivo:** Entender qué features son más importantes.

```python
from enhanced_model import get_feature_importance

# Entrenar Random Forest
model = build_pipeline("rf")
model.fit(X_train, y_train)

# Extraer importancia
feature_names = [f"delta_{i}" for i in range(20)] + ["mean", "std", "skew"]
importances = get_feature_importance(model, feature_names)

# Visualizar top 10
import matplotlib.pyplot as plt

top_10 = importances.head(10)
plt.barh(top_10['feature'], top_10['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Features')
plt.tight_layout()
plt.show()

# ¿Pregunta?: ¿Son las estadísticas más importantes que los deltas?
```

### Ejercicio 6: Experimento Completo

**Objetivo:** Diseñar y ejecutar un experimento completo.

**Hipótesis:** "Aumentar el window size mejora la performance"

```python
window_sizes = [10, 20, 30, 50, 100]
results = []

for window in window_sizes:
    X, y, groups = make_windows_from_walks_enhanced(
        positions,
        window=window
    )
    
    X_train, X_test, y_train, y_test, _, _ = group_train_test_split(
        X, y, groups, test_size=0.2
    )
    
    model = build_pipeline("hgb")
    model.fit(X_train, y_train)
    
    metrics = evaluate(model, X_test, y_test)
    results.append({
        'window': window,
        'roc_auc': metrics['roc_auc'],
        'n_samples': len(X)
    })
    print(f"Window {window}: ROC-AUC = {metrics['roc_auc']:.3f}, "
          f"Muestras = {len(X)}")

# Visualizar
import pandas as pd
df = pd.DataFrame(results)
df.plot(x='window', y='roc_auc', marker='o')
plt.xlabel('Window Size')
plt.ylabel('ROC-AUC')
plt.title('Performance vs Window Size')
plt.grid(True)
plt.show()

# ¿Conclusión?: ¿Cuál es el window size óptimo?
```

---

## 📚 Recursos Adicionales

### Libros Recomendados

1. **"Introduction to Statistical Learning"** (James et al.)
   - Capítulo 4: Clasificación
   - Capítulo 8: Tree-Based Methods
   - **Gratis online:** https://www.statlearning.com/

2. **"Pattern Recognition and Machine Learning"** (Bishop)
   - Capítulo 4: Linear Models for Classification

3. **"Hands-On Machine Learning"** (Géron)
   - Excelente para scikit-learn

### Cursos Online

1. **Andrew Ng - Machine Learning (Coursera)**
   - Fundamentos sólidos
   - Gratis para auditar

2. **Fast.ai - Practical Deep Learning**
   - Enfoque práctico

3. **Kaggle Learn**
   - Tutoriales interactivos gratis

### Papers Científicos

1. **Random Walks:**
   - Pearson (1905): "The Problem of the Random Walk"
   - Feller (1968): "An Introduction to Probability Theory"

2. **Financial Applications:**
   - Lo & MacKinlay (1999): "A Non-Random Walk Down Wall Street"
   - Malkiel (2015): "A Random Walk Down Wall Street"

3. **Machine Learning:**
   - Breiman (2001): "Random Forests"
   - Friedman (2001): "Greedy Function Approximation: A Gradient Boosting Machine"

### Documentación Oficial

1. **scikit-learn:**
   - User Guide: https://scikit-learn.org/stable/user_guide.html
   - API Reference: https://scikit-learn.org/stable/modules/classes.html

2. **NumPy/SciPy:**
   - NumPy: https://numpy.org/doc/
   - SciPy: https://docs.scipy.org/

3. **Matplotlib:**
   - Tutorials: https://matplotlib.org/stable/tutorials/index.html

### Comunidades

1. **Stack Overflow**
   - Tag: `scikit-learn`, `machine-learning`

2. **Reddit**
   - r/MachineLearning
   - r/learnmachinelearning
   - r/datascience

3. **Kaggle**
   - Competitions para practicar
   - Notebooks públicos para aprender

### Proyectos Relacionados

1. **Scikit-learn Examples**
   - https://scikit-learn.org/stable/auto_examples/

2. **Kaggle Kernels**
   - Buscar "random walk", "time series classification"

3. **GitHub Topics**
   - #time-series-classification
   - #feature-engineering
   - #gradient-boosting

---

## 🎯 Checklist de Aprendizaje

Marca lo que ya dominas:

### Conceptos Básicos
- [ ] Entiendo qué es un random walk
- [ ] Distingo entre walk justo y sesgado
- [ ] Sé qué es una ventana deslizante
- [ ] Entiendo el concepto de features

### Feature Engineering
- [ ] Puedo explicar raw deltas
- [ ] Entiendo estadísticas (mean, std, skew)
- [ ] Comprendo trend features (slope, correlation)
- [ ] Sé cuándo usar cada tipo de feature

### Modelos
- [ ] Entiendo regresión logística
- [ ] Sé cómo funciona Random Forest
- [ ] Comprendo Gradient Boosting
- [ ] Puedo elegir el modelo adecuado

### Validación
- [ ] Distingo entre accuracy, precision y recall
- [ ] Entiendo ROC-AUC
- [ ] Sé por qué usamos baselines
- [ ] Comprendo el concepto de overfitting

### Group-Aware Validation
- [ ] Entiendo el problema del data leakage
- [ ] Sé implementar group-aware split
- [ ] Uso GroupKFold para CV
- [ ] Puedo explicar por qué es crítico

### Práctico
- [ ] Puedo entrenar un modelo básico
- [ ] Sé interpretar las métricas
- [ ] Puedo comparar modelos
- [ ] Entiendo feature importance
- [ ] Sé diseñar experimentos

---

## 🚀 Próximos Pasos

1. **Practica con los ejercicios** de esta guía
2. **Lee el código** en `enhanced_model.py` línea por línea
3. **Ejecuta el notebook** `random_walk_prediction_fast-Copy1.ipynb`
4. **Experimenta** con tus propios parámetros
5. **Lee los ejemplos** en `examples/` para aplicaciones reales
6. **Contribuye** al proyecto con mejoras

---

## ❓ Preguntas Frecuentes

### ¿Por qué Random Walks?

Los random walks son un modelo simple que aparece en muchos fenómenos reales: precios de acciones, movimiento browniano, procesos de difusión, etc. Si puedes detectar patrones aquí, puedes aplicarlo a problemas reales.

### ¿Es esto "Deep Learning"?

No, este proyecto usa **Machine Learning tradicional** (scikit-learn). Es más simple, interpretable y suficiente para muchos problemas. Deep Learning (LSTM, transformers) se puede añadir después si es necesario.

### ¿Qué ROC-AUC es "bueno"?

Depende del contexto:
- **0.50-0.60:** Señal muy débil, quizás no vale la pena
- **0.60-0.70:** Aceptable para exploración
- **0.70-0.80:** Bueno, útil en producción
- **0.80+:** Excelente, pero verifica que no hay overfitting

### ¿Cómo evito overfitting?

1. ✅ Usa **group-aware validation** (siempre)
2. ✅ Regularización (L2, max_depth, min_samples_leaf)
3. ✅ Cross-validation con múltiples folds
4. ✅ Compara train vs test performance
5. ✅ Usa baselines para verificar

### ¿Puedo usar esto para trading real?

**Con precaución.** Este proyecto es educativo. Para trading real:
- Necesitas muchos más features (fundamentales, sentimiento, etc.)
- Considera costos de transacción
- Implementa risk management
- Backtesting riguroso
- Empieza con paper trading

### ¿Qué más puedo aprender?

- **Deep Learning:** LSTM, GRU para series temporales
- **Online Learning:** Actualizar modelos en tiempo real
- **Ensemble Methods:** Combinar múltiples modelos
- **Feature Selection:** Elegir features automáticamente
- **Hyperparameter Optimization:** Optuna, Hyperopt

---

**¿Tienes más preguntas?** Abre un issue en GitHub o contribuye a esta guía. ¡El aprendizaje es colaborativo! 🎓

---

<div align="center">

**¡Feliz aprendizaje! 🚀**

[⬅️ Volver al README](README.md) • [📊 Ver Ejemplos](examples/) • [🧪 Ejecutar Tests](tests/)

</div>
