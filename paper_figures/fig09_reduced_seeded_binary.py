#!/usr/bin/env python3
"""Fig. 9 (fig:equal_mass_reduced_seeded): reduced-model seeded-dipole equal-mass binary vs the unseeded binary.

Source runs: binary2_reduced_seeded (run ID 20251024_221210@128, n=1 l=1 seeded, N=128, DF A=0.75, BH does not
source the ULDM) and binary256 (unseeded, fully coupled, N=256).
Observable: |x_BH| from the simulation origin, pc.  seeded A = body 0 (BH1), seeded B = body 1 (BH2): in the
digitised source PDF distance_from_origin_0_pc_v3.pdf drawing 17 is the dark-grey BH1 line and drawing 18 the
light-grey dashed BH2 line.  The published curves were digitised from PDFs; this script reads the raw runs.
See _seeded_binary.py for the shared style and options.
"""

from __future__ import annotations

import _seeded_binary


def main(argv=None):
    return _seeded_binary.run_figure("fig09", "binary2_reduced_seeded", 0, 1, __doc__, argv)


if __name__ == "__main__":
    main()
