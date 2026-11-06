"""Die festen Instanzen der Leiter für test_claims.py, parallel gerechnet (ein Prozess je Kern, höchstens vier) und je Größe nur einmal.

Zeiten stehen in den Zeilen, aber nie in den Tests; die Parallelität ändert nur die gemessenen Sekunden."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import sdp_evaluation as E

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
