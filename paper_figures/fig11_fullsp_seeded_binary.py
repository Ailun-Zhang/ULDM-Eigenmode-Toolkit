#!/usr/bin/env python3
"""Fig. 11 (tex label fig:equal_mass_seeded_fullSP): fully coupled seeded-dipole equal-mass binary vs the unseeded binary.

Source runs: binary2_fullsp_seeded (run ID 20251029_030150@128, n=1 l=1 seeded, N=128, fully coupled, no
empirical drag) and binary256 (unseeded, fully coupled, N=256).
Observable: |x_BH| from the simulation origin, pc.  seeded A = body 1 (BH2), seeded B = body 0 (BH1): the
digitised source (thesis figure distance_from_origin_1_pc_v4.pdf) drawing 18 is the red "BH2 (from Origin)"
line of Test_PyUL_alan_Binary.ipynb cell 27 and drawing 25 the black BH1 line.  The published curves were
digitised from PDFs; this script reads the raw runs.  See _seeded_binary.py for the shared style and options.
"""

from __future__ import annotations

import _seeded_binary


def main(argv=None):
    return _seeded_binary.run_figure("fig11", "binary2_fullsp_seeded", 1, 0, __doc__, argv)


if __name__ == "__main__":
    main()
