"""V2 MGF header adapter for ALREADY-FROZEN MultiMS2 scoring implementation.

Only decodes literal [20.0]/[40.0]/[60.0] values accepted by the
header-only metadata gate. All spectrum matching, geometry, test statistics
and candidate selection are the separately frozen originals.
"""
from __future__ import annotations

import audit_multims2_third_source_metadata as source_headers
from audit_multims2_third_source_metadata_v2 import NEW_EXACT_CID_ENERGY

# The original score implementation imports source_headers.eligible, which
# resolves source_headers.ENERGY at runtime. Modifying this decoder *only*
# permits verification of the selected MGF headers. It cannot change prediction
# parameters, query identities or matching tolerance.
source_headers.ENERGY=NEW_EXACT_CID_ENERGY

from benchmark_multims2_third_source_geometry import main

if __name__=="__main__":
    main()
