"""
figure_utils.py — Utilidades para figuras IEEE/A4 — Proyecto WTI Tesis
=======================================================================
Versión  : 1.0
Fecha    : 2026-03-17
Proyecto : Benchmark de modelos de predicción de volatilidad WTI

Compatible con Google Colab (no requiere instalación adicional).

Uso rápido
----------
# Opción A — importar desde Drive en Colab:
    import sys
    sys.path.insert(0, '/content/drive/My Drive/2410VDSO Trabajo de Grado/Modelos/utils')
    from figure_utils import fig_size, save_fig, apply_thesis_style
    apply_thesis_style('A4')

# Opción B — pegar directamente en celda de config de cada notebook.

Funciones exportadas
--------------------
fig_size(layout, fmt, aspect) → (w, h) en pulgadas para impresión en A4/Letter
save_fig(fig, filename, ...)  → guarda con 300dpi y valida dimensiones
apply_thesis_style(fmt_page)  → aplica rcParams IEEE + imprime referencia de tamaños
"""

import os
import warnings
import matplotlib
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTES DE PÁGINA  (en pulgadas)
# =============================================================================

_PAGE_DIM = {
    'A4':     {'w': 8.27,  'h': 11.69},   # ISO 216 — 210 × 297 mm
    'letter': {'w': 8.50,  'h': 11.00},   # ANSI A  — 215.9 × 279.4 mm
}

# Márgenes estándar de tesis en pulgadas
#   izquierda: 3 cm (1.18") — encuadernación
#   resto:     2.5 cm (0.98")
_MARGINS = {
    'top':    0.98,
    'bottom': 0.98,
    'left':   1.18,
    'right':  0.98,
}


def _text_area(fmt: str = 'A4') -> tuple:
    """
    Retorna (ancho, alto) del área de texto disponible en pulgadas
    para el formato de página especificado.

    Parámetros
    ----------
    fmt : 'A4' o 'letter'

    Retorna
    -------
    (text_width, text_height) en pulgadas
        A4:     ≈ (6.11, 9.73)
        letter: ≈ (6.34, 9.04)
    """
    if fmt not in _PAGE_DIM:
        raise ValueError(f"fmt debe ser 'A4' o 'letter'. Recibido: '{fmt}'")
    p = _PAGE_DIM[fmt]
    w = p['w'] - _MARGINS['left'] - _MARGINS['right']
    h = p['h'] - _MARGINS['top']  - _MARGINS['bottom']
    return round(w, 4), round(h, 4)


# =============================================================================
# FUNCIÓN PRINCIPAL: fig_size()
# =============================================================================

def fig_size(layout: str = 'full', fmt: str = 'A4',
             aspect: float = 0.618) -> tuple:
    """
    Retorna figsize (w, h) en pulgadas compatible con impresión en A4 o letter.

    Parámetros
    ----------
    layout : str
        'full'   → ancho completo, alto = ancho × aspect (proporción áurea ≈ 0.618)
        'half'   → mitad del ancho (para 2 figuras lado a lado en LaTeX)
        'wide'   → ancho completo, alto muy comprimido (serie temporal larga)
        '2x1'    → 2 paneles horizontales  (h/w ≈ 0.50)
        '2x2'    → 4 paneles en cuadrícula (h/w ≈ 0.75)
        '3x1'    → 3 paneles horizontales  (h/w ≈ 0.38)
        'square' → cuadrado  (w = h = ancho del área de texto)
        'tall'   → alto extendido ≤ 85 % de la altura de página
        'page'   → ocupa casi toda la página (para figuras resumen)
    fmt    : 'A4' (default) o 'letter'
    aspect : float — proporción alto/ancho para layout='full'.
             Default: 0.618 (proporción áurea inversa)

    Retorna
    -------
    tuple (width, height) en pulgadas para usar en figsize=

    Ejemplos
    --------
    >>> fig, ax  = plt.subplots(figsize=fig_size('full'))       # 6.11 × 3.77"
    >>> fig, axs = plt.subplots(1, 2, figsize=fig_size('2x1')) # 6.11 × 3.06"
    >>> fig      = plt.figure(figsize=fig_size('2x2'))          # 6.11 × 4.58"
    """
    tw, th = _text_area(fmt)

    opts = {
        'full':   (tw,      tw * aspect),
        'half':   (tw / 2,  (tw / 2) * aspect),
        'wide':   (tw,      tw * 0.45),
        '2x1':    (tw,      tw * 0.50),
        '2x2':    (tw,      tw * 0.75),
        '3x1':    (tw,      tw * 0.38),
        'square': (tw,      tw),
        'tall':   (tw,      min(tw * 1.40, th * 0.85)),
        'page':   (tw,      th * 0.90),

        # --- NUEVOS PRESETS NB03 ---
        # 3 paneles verticales con anotaciones (7.1)
        'nb03_ts':   (tw, min(tw * 1.45, th * 0.92)),   # ~6.11 x 8.86 en A4
        # KDE + boxplot (7.2)
        'nb03_dist': (tw, tw * 0.62),                   # ~6.11 x 3.79
        # Heatmap triangular con labels largos (7.3)
        'nb03_corr': (tw, tw * 0.95),                   # ~6.11 x 5.80
    }

    if layout not in opts:
        warnings.warn(
            f"Layout '{layout}' no reconocido. Opciones: {list(opts)}. "
            "Usando 'full'.", UserWarning, stacklevel=2
        )
        layout = 'full'

    return tuple(round(x, 4) for x in opts[layout])


# =============================================================================
# FUNCIÓN DE GUARDADO ESTANDARIZADO: save_fig()
# =============================================================================

def save_fig(fig_or_none, filename: str, output_path: str = '.',
             dpi: int = 300, fmt: str = 'png',
             fmt_page: str = 'A4', warn: bool = True) -> str:
    """
    Guarda una figura matplotlib con parámetros estándar del proyecto.

    Emite UserWarning si las dimensiones de la figura exceden el área de
    texto del formato de página indicado.

    Parámetros
    ----------
    fig_or_none : matplotlib.figure.Figure o None
        Figura a guardar. Si None, utiliza plt.gcf().
    filename    : str
        Nombre del archivo SIN extensión (ej: 'fig_benchmark_f1').
    output_path : str
        Directorio de destino. Se crea automáticamente si no existe.
    dpi         : int
        Resolución de guardado (default 300 — impresión de alta calidad).
    fmt         : str
        Formato de imagen: 'png' (default), 'pdf', 'svg'.
    fmt_page    : str
        Formato de página para validación dimensional: 'A4' o 'letter'.
    warn        : bool
        Si True, emite advertencia cuando la figura excede los límites.

    Retorna
    -------
    str — ruta absoluta del archivo guardado.

    Ejemplos
    --------
    >>> path = save_fig(fig, 'fig_roc_curve', output_path=FIG_PATH)
    >>> path = save_fig(None, 'fig_loss', output_path=FIG_PATH, fmt='pdf')
    """
    fig = fig_or_none if fig_or_none is not None else plt.gcf()
    w, h = fig.get_size_inches()
    tw, th = _text_area(fmt_page)

    # Tolerancia del 5 % para evitar alertas por redondeos de figsize
    if warn and (w > tw * 1.05 or h > th * 0.93):
        warnings.warn(
            f"Figura '{filename}': {w:.2f}\" × {h:.2f}\" excede el área de texto "
            f"{fmt_page} ({tw:.2f}\" × {th:.2f}\"). "
            f"Considera usar fig_size() para obtener dimensiones correctas.",
            UserWarning, stacklevel=2
        )

    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, f"{filename}.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.04, facecolor='white')
    print(f"  [OK] Guardada: {path}  [{w:.2f}\" x {h:.2f}\" @ {dpi} dpi]")
    return path


# =============================================================================
# APLICAR ESTILO IEEE: apply_thesis_style()
# =============================================================================

def apply_thesis_style(fmt_page: str = 'A4') -> None:
    """
    Aplica la configuración IEEE de matplotlib del proyecto WTI y muestra
    la tabla de tamaños de referencia para el formato de página indicado.

    Llamar UNA SOLA VEZ al inicio de cada notebook en la celda de
    configuración global. Reemplaza el bloque inline de
    plt.rcParams.update(...) que ya existe en NB01/NB02/NB03.

    Parámetros
    ----------
    fmt_page : 'A4' (default) o 'letter'

    Ejemplo
    -------
    >>> apply_thesis_style('A4')
    📐 Estilo IEEE aplicado — Área de texto A4: 6.11" × 9.73" (15.5 × 24.7 cm)
       Tamaños de referencia:
         fig_size('full'  ) → (6.11, 3.77)"  = 15.5 × 9.6 cm
         fig_size('2x1'   ) → (6.11, 3.06)"  = 15.5 × 7.8 cm
         ...
    """
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid')   # fallback versiones antiguas

    plt.rcParams.update({
        # Fuente IEEE-style
        'font.family':      'serif',
        'font.serif':       ['Times New Roman', 'DejaVu Serif'],
        'font.size':        10,
        'axes.titlesize':   11,
        'axes.titleweight': 'bold',
        'axes.labelsize':   11,
        'legend.fontsize':  9,
        # Resolución
        'figure.dpi':       150,    # visualización en notebook (no impresión)
        'savefig.dpi':      300,    # guardado siempre a 300 dpi
        # Cuadrícula
        'axes.grid':        True,
        'grid.alpha':       0.25,
        'grid.linestyle':   ':',
        # Ejes
        'axes.edgecolor':   'black',
        'axes.linewidth':   0.8,
            'axes.titlepad':      8,
        'lines.linewidth':    1.4,
        'legend.frameon':     False,
        'legend.borderaxespad': 0.4,
        'xtick.major.pad':    3.5,
        'ytick.major.pad':    3.5,
    })

    tw, th = _text_area(fmt_page)
    print(f"[IEEE] Estilo aplicado -- Area de texto {fmt_page}: "
          f"{tw:.2f}\" x {th:.2f}\"  ({tw * 2.54:.1f} x {th * 2.54:.1f} cm)")
    print("   Tamanos de referencia (figsize en pulgadas):")
    for lyt in ['full', 'half', 'wide', '2x1', '2x2', '3x1', 'tall', 'page']:
        ww, hh = fig_size(lyt, fmt_page)
        print(f"     fig_size('{lyt:<6}') -> ({ww:.2f}, {hh:.2f})\"  "
              f"= {ww * 2.54:.1f} x {hh * 2.54:.1f} cm")


# =============================================================================
# CELDA DE IMPORT PARA NOTEBOOKS  (copiar en la sección de configuración)
# =============================================================================
# El bloque de abajo está pensado para pegarse directamente en un notebook.
# No se ejecuta al importar este módulo.

_NOTEBOOK_IMPORT_CELL = r'''
# ── Importar utilidades de figura (auto-detecta Colab / local) ──────────────
import sys, os

_UTILS_COLAB = '/content/drive/My Drive/2410VDSO Trabajo de Grado/Modelos/utils'
_UTILS_LOCAL = os.path.join(os.path.dirname(os.path.abspath('.')), 'utils')

for _p in [_UTILS_COLAB, _UTILS_LOCAL]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from figure_utils import fig_size, save_fig, apply_thesis_style
    apply_thesis_style('A4')   # aplica rcParams IEEE y muestra tabla de tamaños
except ImportError:
    # ── Fallback inline ─────────────────────────────────────────────────────
    import warnings
    def fig_size(layout='full', fmt='A4', aspect=0.618):
        tw = 6.11 if fmt == 'A4' else 6.34
        th = 9.73 if fmt == 'A4' else 9.04
        opts = {
            'full'  : (tw,      tw * aspect),
            'half'  : (tw / 2,  (tw / 2) * aspect),
            'wide'  : (tw,      tw * 0.45),
            '2x1'   : (tw,      tw * 0.50),
            '2x2'   : (tw,      tw * 0.75),
            '3x1'   : (tw,      tw * 0.38),
            'square': (tw,      tw),
            'tall'  : (tw,      min(tw * 1.40, th * 0.85)),
            'page'  : (tw,      th * 0.90),
        }
        return opts.get(layout, (tw, tw * aspect))

    def save_fig(fig, name, output_path='.', dpi=300, fmt='png', **kw):
        os.makedirs(output_path, exist_ok=True)
        p = os.path.join(output_path, f"{name}.{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Guardada: {p}")
        return p

    print("[WARN] figure_utils.py no encontrado -- usando funciones fallback inline.")
    print("   Asegurate de subir utils/figure_utils.py a Google Drive.")
'''

# =============================================================================
# __all__ — símbolos exportados
# =============================================================================

__all__ = ['fig_size', 'save_fig', 'apply_thesis_style']


# =============================================================================
# TEST RÁPIDO  (ejecutar como script: python figure_utils.py)
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TEST figure_utils.py")
    print("=" * 60)

    # 1. Probar _text_area
    print("\n--- _text_area ---")
    for f in ('A4', 'letter'):
        tw, th = _text_area(f)
        print(f"  {f:6s}: {tw:.4f}\" × {th:.4f}\"  ({tw*2.54:.2f} × {th*2.54:.2f} cm)")

    # 2. Probar fig_size
    print("\n--- fig_size ---")
    apply_thesis_style('A4')

    # 3. Crear y guardar figura de prueba
    print("\n--- save_fig (figura de prueba) ---")
    import numpy as np
    fig, ax = plt.subplots(figsize=fig_size('full'))
    x = np.linspace(0, 2 * np.pi, 200)
    ax.plot(x, np.sin(x), label='sin(x)')
    ax.plot(x, np.cos(x), label='cos(x)', ls='--')
    ax.set_title('Figura de prueba — fig_size(full) — A4')
    ax.legend()
    fig.tight_layout()
    p = save_fig(fig, 'test_fig_size', output_path='.', dpi=150)
    plt.close(fig)
    print(f"\n  Figura de prueba guardada en: {p}")
    print("\n[PASS] Todos los tests pasaron correctamente.")
