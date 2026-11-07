"""Die festen Instanzen der Leiter für test_claims.py, parallel gerechnet (ein Prozess je Kern, höchstens vier) und je Größe nur einmal.

Zeiten stehen in den Zeilen, aber nie in den Tests; die Parallelität ändert nur die gemessenen Sekunden."""

import multiprocessing
import os

# Ein BLAS-Thread je Prozess, bevor numpy geladen wird (die Kindprozesse erben die Umgebung): mehrere Prozesse mit mehrfädigem BLAS auf wenigen Kernen
# verlangsamen die kleinen Eigenzerlegungen des ADMM um ein Vielfaches (Linux-CI: Zeilen für n = 12 über zweieinhalb Stunden statt Minuten).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import sdp_evaluation as E  # noqa: E402

_ROWS: dict = {}


def _workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _job(job):
    return E.ladder_row(*job)


def ladder_rows(n: int) -> list:
    """Alle Zeilen der Größe `n` in der Reihenfolge von `E.ladder_jobs` (sechs Familien)."""
    if n not in _ROWS:
        with ProcessPoolExecutor(max_workers=_workers(), mp_context=multiprocessing.get_context("spawn")) as ex:
            _ROWS[n] = list(ex.map(_job, E.ladder_jobs(n), chunksize=1))
    return _ROWS[n]
