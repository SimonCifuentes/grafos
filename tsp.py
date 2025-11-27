"""
============================================================
Problema del Viajante (TSP) sobre un grafo completo de
ciudades chilenas – Versión documentada
============================================================

Este script resuelve una instancia pequeña del Problema del
Viajante (TSP) definida por 7 ciudades reales de Chile.
Se comparan dos enfoques:

1) Búsqueda exhaustiva (método exacto)
2) Heurística constructiva de Vecino Más Cercano (Nearest Neighbor)

El programa:

- Imprime en consola:
    * Planteamiento del problema
    * Tabla de ciudades y coordenadas
    * Matriz de distancias D
    * Ruta óptima (exhaustiva) y su longitud L_opt
    * Ruta heurística (NN) y su longitud L_NN
    * Tiempos de ejecución de ambos métodos
    * Gap de optimalidad y comentario cualitativo

- Genera figuras estáticas (carpeta ./figuras):
    * mapa_ciudades.png
    * ruta_optima_exhaustiva.png
    * ruta_heuristica_nn.png

- Abre un visor interactivo con Matplotlib:
    * Modo “Exhaustivo”: recorre los ciclos evaluados, mostrando
      también el mejor ciclo encontrado hasta el momento.
    * Modo “Vecino más cercano”: muestra la ruta heurística
      paso a paso y explica la decisión en cada iteración.

Requisitos:
    - Python 3.10+ (probado en 3.12)
    - matplotlib

Ejecución:
    python tsp_documentado.py
"""

from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons


# -------------------------------------------------------------
# 1. Datos de entrada: ciudades y coordenadas
# -------------------------------------------------------------


@dataclass
class Ciudad:
    """Representa una ciudad con nombre y coordenadas geográficas."""

    nombre: str
    latitud: float
    longitud: float


# Lista de ciudades de la instancia (puedes modificarla si quieres
# experimentar con otras ciudades o con más nodos).
CIUDADES: List[Ciudad] = [
    Ciudad("Santiago", -33.4489, -70.6693),
    Ciudad("Valparaiso", -33.0472, -71.6127),
    Ciudad("Rancagua", -34.1708, -70.7444),
    Ciudad("Talca", -35.4264, -71.6554),
    Ciudad("Chillan", -36.6066, -72.1034),
    Ciudad("Concepcion", -36.8270, -73.0503),
    Ciudad("Temuco", -38.7359, -72.5904),
]

# Cantidad de ciudades de la instancia
NUM_CIUDADES: int = len(CIUDADES)

# Para graficar usamos (longitud, latitud) como (x, y)
COORDS: List[Tuple[float, float]] = [
    (c.longitud, c.latitud) for c in CIUDADES
]


# -------------------------------------------------------------
# 2. Distancias y matriz D
# -------------------------------------------------------------


def distancia_2d(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos 2D.

    Parámetros
    ----------
    a, b:
        Puntos en el plano (x, y).

    Retorna
    -------
    float
        Distancia euclidiana entre a y b.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def construir_matriz_distancias(
    coords: Sequence[Tuple[float, float]],
) -> List[List[float]]:
    """
    Construye la matriz de distancias D[i][j] entre todas las ciudades.

    Parámetros
    ----------
    coords:
        Lista de coordenadas (x, y) de cada ciudad.

    Retorna
    -------
    List[List[float]]
        Matriz de distancias D de tamaño n x n.
    """
    n = len(coords)
    matriz: List[List[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                matriz[i][j] = distancia_2d(coords[i], coords[j])

    return matriz


def longitud_ciclo(
    tour: Sequence[int],
    D: List[List[float]],
) -> float:
    """
    Calcula la longitud total de un ciclo Hamiltoniano cerrado.

    El tour se asume como una lista de índices de ciudades que incluye
    el regreso al origen, por ejemplo: [0, 1, 2, 0].

    Parámetros
    ----------
    tour:
        Secuencia de índices de ciudades.
    D:
        Matriz de distancias D[i][j].

    Retorna
    -------
    float
        Longitud total del ciclo.
    """
    total = 0.0
    for k in range(len(tour) - 1):
        i, j = tour[k], tour[k + 1]
        total += D[i][j]
    return total


def imprimir_instancia() -> None:
    """
    Imprime en consola la tabla de ciudades y sus coordenadas.

    Esta salida se puede copiar fácilmente al informe.
    """
    print("------------------------------------------------------------")
    print("1) Presentacion de la instancia del problema")
    print("------------------------------------------------------------")
    print("Conjunto de ciudades seleccionadas (tabla de coordenadas):\n")
    print(f"{'Indice':>6}  {'Ciudad':<12}  {'Latitud':>10}  {'Longitud':>10}")
    print("-" * 46)
    for idx, c in enumerate(CIUDADES):
        print(f"{idx:>6}  {c.nombre:<12}  {c.latitud:10.4f}  {c.longitud:10.4f}")
    print("\nFuente: Google Maps, consulta realizada el 21 de noviembre de 2025.\n")


def imprimir_matriz_distancias(D: List[List[float]]) -> None:
    """
    Imprime en consola la matriz de distancias D con cuatro decimales.

    Esta salida también se puede usar como base para una tabla en el informe.
    """
    print("------------------------------------------------------------")
    print("2) Matriz de distancias D (distancia euclidiana en grados)")
    print("------------------------------------------------------------")

    # Usamos nombres abreviados para que la tabla sea legible
    nombres_cortos = [c.nombre[:6].ljust(8) for c in CIUDADES]
    encabezado = " " * 10 + " ".join(n.ljust(10) for n in nombres_cortos)
    print(encabezado)

    for i, fila in enumerate(D):
        fila_str = " ".join(f"{d:10.4f}" for d in fila)
        print(nombres_cortos[i].ljust(10) + fila_str)
    print()


# -------------------------------------------------------------
# 3. Método exacto: búsqueda exhaustiva (con historia)
# -------------------------------------------------------------


def tsp_exhaustivo_con_historia(
    D: List[List[float]],
    origen: int = 0,
) -> Tuple[Tuple[int, ...], float, float, List[Dict[str, Any]]]:
    """
    Resuelve el TSP mediante búsqueda exhaustiva y almacena la historia
    de la búsqueda para poder animarla.

    Recorre todas las permutaciones de las ciudades excepto la de origen.
    Para cada permutación se construye el ciclo cerrado (origen, ..., origen)
    y se calcula su longitud. Se mantiene siempre el mejor ciclo encontrado.

    Parámetros
    ----------
    D:
        Matriz de distancias D[i][j].
    origen:
        Índice de la ciudad de origen (por defecto 0).

    Retorna
    -------
    mejor_tour:
        Tupla con el ciclo óptimo (incluyendo el regreso al origen).
    mejor_longitud:
        Longitud del ciclo óptimo.
    tiempo:
        Tiempo de ejecución en segundos.
    historia:
        Lista de diccionarios con información de cada ciclo evaluado:
        - 'iter'        : número de iteración
        - 'tour'        : ciclo actual
        - 'length'      : longitud del ciclo actual
        - 'best_tour'   : mejor ciclo hasta el momento
        - 'best_length' : longitud del mejor ciclo
    """
    n = len(D)
    otras_ciudades = [i for i in range(n) if i != origen]

    mejor_tour: Tuple[int, ...] | None = None
    mejor_longitud: float = float("inf")
    historia: List[Dict[str, Any]] = []

    inicio = time.perf_counter()
    total_rutas = 0

    for iteracion, perm in enumerate(itertools.permutations(otras_ciudades), start=1):
        tour = (origen,) + perm + (origen,)
        longitud_actual = longitud_ciclo(tour, D)
        total_rutas += 1

        if longitud_actual < mejor_longitud:
            mejor_longitud = longitud_actual
            mejor_tour = tour

        historia.append(
            {
                "iter": iteracion,
                "tour": tour,
                "length": longitud_actual,
                "best_tour": mejor_tour,
                "best_length": mejor_longitud,
            }
        )

    tiempo = time.perf_counter() - inicio

    print("------------------------------------------------------------")
    print("3) Resultados del metodo exacto: Busqueda exhaustiva")
    print("------------------------------------------------------------")
    print(f"Numero de rutas evaluadas: {total_rutas}")
    print(f"Ruta optima (indices): {mejor_tour}")
    nombres_ruta = " -> ".join(CIUDADES[i].nombre for i in mejor_tour)
    print(f"Ruta optima (ciudades): {nombres_ruta}")
    print(f"Longitud optima L_opt  = {mejor_longitud:.4f}")
    print(f"Tiempo de ejecucion    = {tiempo:.4f} s\n")

    return mejor_tour, mejor_longitud, tiempo, historia


# -------------------------------------------------------------
# 4. Heurística: Vecino Más Cercano (con historia)
# -------------------------------------------------------------


def tsp_vecino_mas_cercano_con_historia(
    D: List[List[float]],
    origen: int = 0,
) -> Tuple[Tuple[int, ...], float, float, List[Dict[str, Any]]]:
    """
    Aplica la heurística del Vecino Más Cercano y registra una historia
    detallada del proceso para animación y análisis.

    En cada paso:
        - desde la ciudad actual se calculan las distancias a todas las
          ciudades no visitadas,
        - se elige la más cercana,
        - se actualiza la ruta parcial.

    Parámetros
    ----------
    D:
        Matriz de distancias.
    origen:
        Índice de la ciudad de origen.

    Retorna
    -------
    tour_final:
        Ciclo completo (incluyendo regreso al origen).
    longitud_final:
        Longitud del ciclo resultante.
    tiempo:
        Tiempo de ejecución en segundos.
    historia:
        Lista de diccionarios con el detalle de cada paso:
        - 'step'           : número de paso (0 = solo origen)
        - 'tour_partial'   : ruta parcial (cerrada al origen para medir L)
        - 'current'        : ciudad actual
        - 'chosen'         : ciudad elegida en el paso (None en paso 0)
        - 'length_partial' : longitud del ciclo parcial
        - 'candidates'     : lista ordenada de (indice_ciudad, distancia)
    """
    n = len(D)
    no_visitadas = set(range(n))
    no_visitadas.remove(origen)

    ruta: List[int] = [origen]
    ciudad_actual: int = origen

    historia: List[Dict[str, Any]] = []

    # Paso 0: solo la ciudad de origen
    historia.append(
        {
            "step": 0,
            "tour_partial": tuple(ruta),
            "current": origen,
            "chosen": None,
            "length_partial": 0.0,
            "candidates": [],
        }
    )

    inicio = time.perf_counter()
    paso = 0

    while no_visitadas:
        # Posibles candidatas ordenadas por distancia
        candidatos = sorted(
            ((j, D[ciudad_actual][j]) for j in no_visitadas),
            key=lambda x: x[1],
        )
        ciudad_siguiente = candidatos[0][0]

        ruta.append(ciudad_siguiente)
        no_visitadas.remove(ciudad_siguiente)
        paso += 1

        # Cerramos al origen para medir la longitud parcial
        ruta_parcial = ruta + [origen]
        longitud_parcial = longitud_ciclo(ruta_parcial, D)

        historia.append(
            {
                "step": paso,
                "tour_partial": tuple(ruta_parcial),
                "current": ciudad_actual,
                "chosen": ciudad_siguiente,
                "length_partial": longitud_parcial,
                "candidates": candidatos,
            }
        )

        ciudad_actual = ciudad_siguiente

    # Cerramos ciclo definitivo
    ruta.append(origen)
    longitud_final = longitud_ciclo(ruta, D)
    tiempo = time.perf_counter() - inicio

    print("------------------------------------------------------------")
    print("4) Resultados de la heuristica: Vecino Mas Cercano (NN)")
    print("------------------------------------------------------------")
    print(f"Ruta heuristica (indices): {tuple(ruta)}")
    nombres_ruta = " -> ".join(CIUDADES[i].nombre for i in ruta)
    print(f"Ruta heuristica (ciudades): {nombres_ruta}")
    print(f"Longitud L_NN            = {longitud_final:.4f}")
    print(f"Tiempo de ejecucion      = {tiempo:.6f} s\n")

    # Detalle textual paso a paso (útil para el informe o para entender NN)
    print("Detalle paso a paso del Vecino Mas Cercano:")
    for estado in historia:
        step = estado["step"]
        if step == 0:
            print("  Paso 0: se parte en", CIUDADES[origen].nombre)
            continue

        tour_txt = " -> ".join(
            CIUDADES[i].nombre for i in estado["tour_partial"]
        )
        chosen = estado["chosen"]
        current = estado["current"]
        dist_elegida = D[current][chosen]

        print(
            f"  Paso {step}: desde {CIUDADES[current].nombre} "
            f"se elige {CIUDADES[chosen].nombre} (dist = {dist_elegida:.4f}). "
            f"Ruta parcial: {tour_txt}  | L = {estado['length_partial']:.4f}"
        )
    print()

    return tuple(ruta), longitud_final, tiempo, historia


# -------------------------------------------------------------
# 5. Gap de optimalidad y comparación de métodos
# -------------------------------------------------------------


def calcular_gap(L_opt: float, L_nn: float) -> float:
    """
    Calcula el gap de optimalidad en porcentaje.

    g = (L_nn - L_opt) / L_opt * 100
    """
    return (L_nn - L_opt) / L_opt * 100.0


def imprimir_comparacion(
    L_opt: float,
    L_nn: float,
    t_opt: float,
    t_nn: float,
) -> None:
    """
    Imprime una tabla comparativa y comentarios sobre los métodos.

    Parámetros
    ----------
    L_opt:
        Longitud del ciclo óptimo (búsqueda exhaustiva).
    L_nn:
        Longitud del ciclo resultante de NN.
    t_opt:
        Tiempo de ejecución del método exhaustivo.
    t_nn:
        Tiempo de ejecución de NN.
    """
    gap = calcular_gap(L_opt, L_nn)
    factor_tiempo = t_opt / t_nn if t_nn > 0 else float("inf")

    print("------------------------------------------------------------")
    print("5) Comparacion cuantitativa de metodos")
    print("------------------------------------------------------------")
    print(f"{'Metodo':<20} {'Longitud':>10} {'Tiempo [s]':>12}")
    print("-" * 44)
    print(f"{'Exhaustivo':<20} {L_opt:10.4f} {t_opt:12.4f}")
    print(f"{'Vecino Mas Cercano':<20} {L_nn:10.4f} {t_nn:12.6f}")
    print("-" * 44)
    print("Gap de optimalidad g = ((L_NN - L_opt)/L_opt) * 100")
    print(f"g = {gap:.2f} %")
    print(f"Relacion de tiempos (exhaustivo / NN) = {factor_tiempo:.1f} veces\n")

    print("Comentario interpretativo:")
    if gap < 1.0:
        print(
            "  • La heuristica NN produce una ruta practicamente tan buena "
            "como el optimo (gap menor al 1 %)."
        )
    elif gap < 10.0:
        print(
            "  • La solucion heuristica es razonablemente cercana al optimo, "
            "aunque con una diferencia apreciable en la longitud."
        )
    else:
        print(
            "  • La ruta heuristica se aleja de manera significativa del optimo; "
            "en esta instancia NN no es especialmente precisa."
        )

    if factor_tiempo > 100.0:
        print(
            "  • En terminos de tiempo de computo, la busqueda exhaustiva es "
            "mucho mas lenta que NN para esta instancia."
        )
    else:
        print(
            "  • Para este tamano de problema ambos metodos tienen tiempos comparables, "
            "pero el crecimiento factorial del exhaustivo lo vuelve poco escalable "
            "para valores mayores de n."
        )
    print()


# -------------------------------------------------------------
# 6. Funciones de graficado (figuras estáticas)
# -------------------------------------------------------------


def asegurar_directorio(path: str = "figuras") -> str:
    """
    Crea el directorio indicado si no existe y retorna la ruta.

    Se usa para centralizar la ubicación de las figuras que se
    insertarán en el informe.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def graficar_ciudades(
    coords: Sequence[Tuple[float, float]],
    nombre_archivo: str | None = None,
) -> None:
    """
    Genera una figura con la ubicación de las ciudades en el plano.

    Parámetros
    ----------
    coords:
        Lista de coordenadas (x, y).
    nombre_archivo:
        Si se indica una ruta, guarda la figura en disco; en caso
        contrario, la muestra en pantalla.
    """
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]

    plt.figure()
    plt.scatter(xs, ys)

    for i, ciudad in enumerate(CIUDADES):
        plt.text(
            coords[i][0],
            coords[i][1],
            ciudad.nombre,
            fontsize=9,
            ha="right",
            va="bottom",
        )

    plt.title("Ubicacion geografica de las ciudades")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.axis("equal")
    plt.grid(True)

    if nombre_archivo:
        plt.savefig(nombre_archivo, dpi=300, bbox_inches="tight")
        print(f"Figura guardada en {nombre_archivo}")
        plt.close()
    else:
        plt.show()


def graficar_ciclo(
    coords: Sequence[Tuple[float, float]],
    tour: Sequence[int],
    titulo: str,
    nombre_archivo: str | None = None,
) -> None:
    """
    Grafica un ciclo completo sobre las coordenadas geográficas.

    Parámetros
    ----------
    coords:
        Lista de coordenadas de las ciudades.
    tour:
        Secuencia de índices que define el ciclo.
    titulo:
        Título de la figura.
    nombre_archivo:
        Ruta de salida de la imagen, si se desea guardar.
    """
    xs = [coords[i][0] for i in tour]
    ys = [coords[i][1] for i in tour]

    plt.figure()
    plt.scatter([c[0] for c in coords], [c[1] for c in coords])

    for i, ciudad in enumerate(CIUDADES):
        plt.text(
            coords[i][0],
            coords[i][1],
            ciudad.nombre,
            fontsize=9,
            ha="right",
            va="bottom",
        )

    plt.plot(xs, ys, marker="o")
    plt.title(titulo)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.axis("equal")
    plt.grid(True)

    if nombre_archivo:
        plt.savefig(nombre_archivo, dpi=300, bbox_inches="tight")
        print(f"Figura guardada en {nombre_archivo}")
        plt.close()
    else:
        plt.show()


# -------------------------------------------------------------
# 7. Visualización interactiva (ambos métodos)
# -------------------------------------------------------------


class VisualizadorInteractivo:
    """
    Visor interactivo para los dos métodos del TSP.

    - Modo ``Exhaustivo``:
        Muestra el ciclo que se está evaluando en la iteración actual y
        el mejor ciclo encontrado hasta ese momento (línea discontinua).

    - Modo ``Vecino mas cercano``:
        Muestra la construcción paso a paso de la ruta NN. El panel de
        texto explica qué ciudad se elige, cuál es su distancia y qué
        otras alternativas existían.

    El panel descriptivo se coloca a la derecha para no tapar el grafo.
    """

    def __init__(
        self,
        coords: Sequence[Tuple[float, float]],
        historia_exh: List[Dict[str, Any]],
        historia_nn: List[Dict[str, Any]],
        ruta_opt: Sequence[int],
        L_opt: float,
        ruta_nn: Sequence[int],
        L_nn: float,
    ) -> None:
        self.coords = coords
        self.hist_exh = historia_exh
        self.hist_nn = historia_nn
        self.ruta_opt = ruta_opt
        self.L_opt = L_opt
        self.ruta_nn = ruta_nn
        self.L_nn = L_nn

        # Estado actual del visor
        self.algoritmo = "Exhaustivo"  # o "Vecino mas cercano"
        self.indice = 0

        # Figura principal con espacio para texto a la derecha
        self.fig, self.ax = plt.subplots(figsize=(11, 6))
        plt.subplots_adjust(left=0.07, right=0.58, bottom=0.20, top=0.90)

        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        self.ax.scatter(xs, ys)
        for i, ciudad in enumerate(CIUDADES):
            self.ax.text(
                coords[i][0],
                coords[i][1],
                ciudad.nombre,
                fontsize=9,
                ha="right",
                va="bottom",
            )

        self.ax.set_xlabel("Longitud")
        self.ax.set_ylabel("Latitud")
        self.ax.set_title("TSP sobre ciudades de Chile")

        # Líneas de las rutas (actual y mejor)
        (self.linea_ruta,) = self.ax.plot([], [], "-o")
        (self.linea_mejor,) = self.ax.plot([], [], "--", alpha=0.6)

        # Panel de texto (eje sin bordes)
        self.ax_info = self.fig.add_axes([0.60, 0.20, 0.38, 0.70])
        self.ax_info.axis("off")
        self.texto_info = self.ax_info.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            fontsize=9,
            wrap=True,
        )

        # Controles inferiores
        axprev = plt.axes([0.07, 0.05, 0.15, 0.07])
        axnext = plt.axes([0.27, 0.05, 0.15, 0.07])
        axfast = plt.axes([0.47, 0.05, 0.18, 0.07])
        axreset = plt.axes([0.70, 0.05, 0.15, 0.07])
        axradio = plt.axes([0.60, 0.02, 0.30, 0.10])

        self.bprev = Button(axprev, "⟵ Anterior")
        self.bnext = Button(axnext, "Siguiente ⟶")
        self.bfast = Button(axfast, "Rapido +10")
        self.breset = Button(axreset, "Reiniciar")
        self.radio = RadioButtons(axradio, ("Exhaustivo", "Vecino mas cercano"))

        self.bprev.on_clicked(self._anterior)
        self.bnext.on_clicked(self._siguiente)
        self.bfast.on_clicked(self._rapido)
        self.breset.on_clicked(self._reiniciar)
        self.radio.on_clicked(self._cambiar_algoritmo)

        # Dibujamos el primer estado
        self._actualizar()

    # ----- Callbacks de los botones -----

    def _anterior(self, event) -> None:  # noqa: ARG002
        """Retrocede una iteración/paso."""
        if self.indice > 0:
            self.indice -= 1
        self._actualizar()

    def _siguiente(self, event) -> None:  # noqa: ARG002
        """Avanza una iteración/paso."""
        max_index = (
            len(self.hist_exh) - 1
            if self.algoritmo == "Exhaustivo"
            else len(self.hist_nn) - 1
        )
        if self.indice < max_index:
            self.indice += 1
        self._actualizar()

    def _rapido(self, event) -> None:  # noqa: ARG002
        """Salta varias iteraciones/pasos (avance rápido)."""
        salto = 10
        max_index = (
            len(self.hist_exh) - 1
            if self.algoritmo == "Exhaustivo"
            else len(self.hist_nn) - 1
        )
        self.indice = min(self.indice + salto, max_index)
        self._actualizar()

    def _reiniciar(self, event) -> None:  # noqa: ARG002
        """Vuelve al inicio de la animación."""
        self.indice = 0
        self._actualizar()

    def _cambiar_algoritmo(self, label: str) -> None:
        """
        Cambia entre los modos 'Exhaustivo' y 'Vecino mas cercano'.
        """
        self.algoritmo = label
        self.indice = 0
        self._actualizar()

    # ----- Lógica para actualizar la figura -----

    def _actualizar(self) -> None:
        """
        Redibuja la figura según el algoritmo y el índice actuales.

        Dependiendo del modo, se actualizan:
          - las líneas del ciclo actual y del mejor ciclo (exhaustivo),
          - la ruta parcial (vecino más cercano),
          - el texto explicativo en el panel lateral.
        """
        if self.algoritmo == "Exhaustivo":
            estado = self.hist_exh[self.indice]
            tour = estado["tour"]
            L = estado["length"]
            best_tour = estado["best_tour"]
            best_L = estado["best_length"]
            iteracion = estado["iter"]

            xs = [self.coords[i][0] for i in tour]
            ys = [self.coords[i][1] for i in tour]
            self.linea_ruta.set_data(xs, ys)

            xs_best = [self.coords[i][0] for i in best_tour]
            ys_best = [self.coords[i][1] for i in best_tour]
            self.linea_mejor.set_data(xs_best, ys_best)

            total_iters = len(self.hist_exh)
            self.ax.set_title(
                f"Busqueda exhaustiva – ciclo {iteracion}/{total_iters}"
            )

            ruta_txt = " -> ".join(CIUDADES[i].nombre for i in tour)
            best_txt = " -> ".join(CIUDADES[i].nombre for i in best_tour)

            info = (
                "MODO: Busqueda exhaustiva (exacta)\n"
                f"Iteracion actual: {iteracion} de {total_iters}\n\n"
                f"Ciclo actual:\n  {ruta_txt}\n"
                f"L(ciclo actual) = {L:.4f}\n\n"
                f"Mejor ciclo hasta ahora:\n  {best_txt}\n"
                f"L_mejor = {best_L:.4f}\n\n"
                "La animacion muestra como el metodo evalua sistematicamente\n"
                "todos los ciclos posibles y va actualizando el mejor encontrado\n"
                "hasta converger al optimo global."
            )

        else:  # Vecino más cercano
            estado = self.hist_nn[self.indice]
            tour_parcial = estado["tour_partial"]
            L_parcial = estado["length_partial"]
            step = estado["step"]
            chosen = estado["chosen"]
            current = estado["current"]
            candidatos = estado["candidates"]

            xs = [self.coords[i][0] for i in tour_parcial]
            ys = [self.coords[i][1] for i in tour_parcial]
            self.linea_ruta.set_data(xs, ys)
            self.linea_mejor.set_data([], [])

            total_steps = len(self.hist_nn) - 1
            self.ax.set_title(
                f"Vecino mas cercano – paso {step}/{total_steps}"
            )

            if step == 0:
                info = (
                    "MODO: Vecino mas cercano (heuristica greedy)\n"
                    f"Paso 0: se inicia en {CIUDADES[current].nombre}.\n\n"
                    "En cada paso siguiente, la heuristica elige la ciudad\n"
                    "NO visitada mas cercana a la ciudad actual.\n"
                    "Use 'Siguiente' para ver como se construye el ciclo,\n"
                    "y observe su caracter miope (solo mira la mejor opcion local)."
                )
            else:
                dist_elegida = candidatos[0][1] if candidatos else 0.0
                otras = [
                    (CIUDADES[j].nombre, d)
                    for (j, d) in candidatos[1:3]
                ]
                otras_txt = ", ".join(
                    f"{nom} (dist = {d:.4f})" for nom, d in otras
                ) or "—"

                ruta_txt = " -> ".join(
                    CIUDADES[i].nombre for i in tour_parcial
                )

                info = (
                    "MODO: Vecino mas cercano (heuristica greedy)\n"
                    f"Paso {step}: ciudad actual = {CIUDADES[current].nombre}\n"
                    f"Se elige {CIUDADES[chosen].nombre} por ser la ciudad\n"
                    f"no visitada mas cercana (dist = {dist_elegida:.4f}).\n"
                    f"Otras alternativas cercanas eran: {otras_txt}.\n\n"
                    "Ruta parcial (cerrada al origen si corresponde):\n"
                    f"  {ruta_txt}\n"
                    f"L(parcial) = {L_parcial:.4f}\n\n"
                    "La heuristica siempre toma la mejor decision local, sin\n"
                    "considerar el efecto global sobre el ciclo completo, lo que\n"
                    "evidencia su naturaleza greedy o miope."
                )

        self.texto_info.set_text(info)
        self.fig.canvas.draw_idle()


# -------------------------------------------------------------
# 8. Programa principal
# -------------------------------------------------------------


def main() -> None:
    """
    Punto de entrada principal del script.

    - Imprime el planteamiento del problema.
    - Muestra la instancia (ciudades) y la matriz de distancias.
    - Ejecuta la búsqueda exhaustiva y la heurística NN.
    - Compara cuantitativamente ambos métodos.
    - Genera figuras estáticas para el informe.
    - Abre el visualizador interactivo.
    """
    print("============================================================")
    print(" PROBLEMA DEL VIAJANTE (TSP) SOBRE GRAFO COMPLETO")
    print("============================================================\n")
    print(
        "Planteamiento general:\n"
        "  Un viajante debe visitar exactamente una vez cada ciudad de un conjunto\n"
        "  dado y regresar al punto de partida, minimizando la distancia total\n"
        "  recorrida. En esta actividad se compara un metodo exacto (busqueda\n"
        "  exhaustiva) con una heuristica constructiva (Vecino Mas Cercano).\n"
    )

    # 1) Descripción de la instancia y matriz de distancias
    imprimir_instancia()
    D = construir_matriz_distancias(COORDS)
    imprimir_matriz_distancias(D)

    # 2) Carpeta de salida para figuras
    carpeta_fig = asegurar_directorio("figuras")

    # Figura con solo las ciudades (mapa base)
    graficar_ciudades(
        COORDS,
        nombre_archivo=os.path.join(carpeta_fig, "mapa_ciudades.png"),
    )

    # 3) Método exacto (con historia para la animación)
    ruta_opt, L_opt, t_opt, historia_exh = tsp_exhaustivo_con_historia(D, origen=0)

    # 4) Heurística NN (con historia para la animación)
    ruta_nn, L_nn, t_nn, historia_nn = tsp_vecino_mas_cercano_con_historia(D, origen=0)

    # 5) Comparación numérica y comentario cualitativo
    imprimir_comparacion(L_opt, L_nn, t_opt, t_nn)

    # 6) Figuras estáticas de las rutas finales (para el informe)
    graficar_ciclo(
        COORDS,
        ruta_opt,
        titulo="Ruta optima (busqueda exhaustiva)",
        nombre_archivo=os.path.join(carpeta_fig, "ruta_optima_exhaustiva.png"),
    )
    graficar_ciclo(
        COORDS,
        ruta_nn,
        titulo="Ruta heuristica (Vecino Mas Cercano)",
        nombre_archivo=os.path.join(carpeta_fig, "ruta_heuristica_nn.png"),
    )

    print("Figuras estaticas generadas en la carpeta:", carpeta_fig)
    print(
        "\nSe abrira la visualizacion interactiva.\n"
        "Use los controles para:\n"
        "  • Cambiar entre 'Exhaustivo' y 'Vecino mas cercano'.\n"
        "  • Avanzar y retroceder paso a paso.\n"
        "  • Saltar rapido con 'Rapido +10'.\n"
        "  • Reiniciar la animacion.\n"
    )

    # 7) Lanzar visor interactivo
    _ = VisualizadorInteractivo(
        COORDS,
        historia_exh,
        historia_nn,
        ruta_opt,
        L_opt,
        ruta_nn,
        L_nn,
    )
    plt.show()


if __name__ == "__main__":
    main()
