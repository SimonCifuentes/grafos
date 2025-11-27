# 📦 **Problema del Viajante (TSP) sobre Ciudades Chilenas**
### *Búsqueda Exhaustiva vs. Vecino Más Cercano (Nearest Neighbor)*
Proyecto para el curso **Teoría de Grafos – INFO1158**  
**Autores:** Simón Cifuentes · Jecar Yáñez  
**Fecha:** Noviembre 2025  

---

## 🧭 **Descripción General**
Este proyecto implementa y compara dos métodos para resolver el **Problema del Viajante (TSP)** sobre un grafo completo construido a partir de **7 ciudades de Chile**.  
Los métodos utilizados son:

1. **Búsqueda exhaustiva** — método exacto que garantiza encontrar la ruta óptima.
2. **Heurística Vecino Más Cercano (Nearest Neighbor)** — estrategia greedy rápida, pero no necesariamente óptima.

Además, el proyecto incluye:

- Construcción de matriz de distancias.
- Generación automática de figuras estáticas.
- **Visualización interactiva** del proceso de ambos algoritmos.
- Análisis cuantitativo: tiempos, longitudes de ruta y *gap de optimalidad*.

---

## 🗺️ **Instancia del Problema**
Se utilizaron las siguientes siete ciudades chilenas, con coordenadas obtenidas desde Google Maps (21/11/2025):

| Ciudad       | Latitud    | Longitud   |
|--------------|------------|------------|
| Santiago     | -33.4489   | -70.6693   |
| Valparaíso   | -33.0472   | -71.6127   |
| Rancagua     | -34.1708   | -70.7444   |
| Talca        | -35.4264   | -71.6554   |
| Chillán      | -36.6066   | -72.1034   |
| Concepción   | -36.8270   | -73.0503   |
| Temuco       | -38.7359   | -72.5904   |

---

## 🔢 **Métodos Implementados**

### ✔️ Búsqueda Exhaustiva
- Evalúa **todas las permutaciones** posibles.
- Para 7 ciudades: (7 − 1)! = 720 rutas evaluadas.
- Siempre encuentra el **óptimo global**.

### ✔️ Heurística Vecino Más Cercano (NN)
- Selecciona la ciudad no visitada más cercana.
- Complejidad: **O(n²)**.
- Se almacena la historia completa del proceso para animación.

---

## 📊 **Resultados Principales**

| Método | Longitud | Tiempo (s) |
|--------|-----------|------------|
| **Exhaustivo** | 12.7566 | ~0.0009 |
| **Vecino Más Cercano** | 14.3487 | ~0.00003 |

**Gap de optimalidad:** ≈ **12.48%**

---

## 🎨 **Visualizaciones Generadas**

En la carpeta `figuras/` se generan automáticamente:

- `mapa_ciudades.png`
- `ruta_optima_exhaustiva.png`
- `ruta_heuristica_nn.png`

Además, se incluye una **visualización interactiva** con:

✔ Avanzar y retroceder  
✔ Salto rápido (+10)  
✔ Cambio entre algoritmos  
✔ Panel explicativo lateral  

---

## ▶️ **Cómo Ejecutar el Proyecto**

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/SimonCifuentes/grafos
cd grafos
```

### 2️⃣ Instalar dependencias
```bash
pip install matplotlib
```

### 3️⃣ Ejecutar
```bash
python tsp.py
```

---

## 📂 **Estructura del Proyecto**
```
grafos/
│── tsp.py
│── figuras/
│   ├── mapa_ciudades.png
│   ├── ruta_optima_exhaustiva.png
│   ├── ruta_heuristica_nn.png
│── README.md
```

---

## 📚 **Referencias**
- Applegate, D., Bixby, R., Chvátal, V., & Cook, W. (2006). *The Traveling Salesman Problem: A Computational Study.*
- Wikipedia: *Travelling Salesman Problem* — https://en.wikipedia.org/wiki/Travelling_salesman_problem
- Material del curso: Teoría de Grafos (INFO1158)
- Código fuente completo: https://github.com/SimonCifuentes/grafos

---

