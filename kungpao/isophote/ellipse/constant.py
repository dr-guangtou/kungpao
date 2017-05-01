"""All the constants used
in the isophote package.
"""

__all__ = [
    'DEFAULT_THRESHOLD', 'WINDOW_HALF_SIZE', 'FIXED_ELLIPSE', 'FAILED_FIT',
    'PI2', 'MAX_EPS', 'MIN_EPS', 'TOO_MANY_FLAGGED', 'DEFAULT_CONVERGENCY',
    'DEFAULT_MINIT', 'DEFAULT_MAXIT', 'DEFAULT_FFLAG', 'DEFAULT_MAXGERR',
    'DEFAULT_EPS', 'DEFAULT_STEP', 'PHI_MAX', 'PHI_MIN', 'NEAREST_NEIGHBOR',
    'BI_LINEAR', 'MEAN', 'MEDIAN', 'NCELL', 'DEFAULT_SCLIP', 'DEFAULT_POS',
    'DEFAULT_SIZE', 'DEFAULT_PA', 'PI', 'TWOPI'
]

# Pi related
PI = 3.141592653589793
TWOPI = 6.283185307179586
PI2 = 1.5707963267948966

# From centerer.py
DEFAULT_THRESHOLD = 0.1
WINDOW_HALF_SIZE = 5

# From ellipse.py
FIXED_ELLIPSE = 4
FAILED_FIT = 5

# From fitter.py
MAX_EPS = 0.95
MIN_EPS = 0.05
TOO_MANY_FLAGGED = 1

DEFAULT_CONVERGENCY = 0.05
DEFAULT_MINIT = 10
DEFAULT_MAXIT = 50
DEFAULT_FFLAG = 0.7
DEFAULT_MAXGERR = 0.5

# From geometry.py
DEFAULT_EPS = 0.2
DEFAULT_STEP = 0.1

# limits for sector angular width
PHI_MAX = 0.2
PHI_MIN = 0.05

# From integrator.py
# integration modes
NEAREST_NEIGHBOR = 'nearest_neighbor'
BI_LINEAR = 'bi-linear'
MEAN = 'mean'
MEDIAN = 'median'

# sqrt(number of cells) in target pixel
NCELL = 8

# From sample.py
DEFAULT_SCLIP = 3.

# From build_test_data.py
DEFAULT_SIZE = 512
DEFAULT_POS = int(DEFAULT_SIZE / 2)
DEFAULT_PA = 0.
