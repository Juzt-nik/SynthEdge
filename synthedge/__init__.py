"""
SynthEdge — Diagnosis-first synthetic data augmentation.

Usage
-----
    from synthedge import SynthEdge
    from synthedge.transfer import find_matching_gaps, transfer_samples

    # Single dataset
    se = SynthEdge(df, target_col="target")
    report   = se.analyze()
    aug_df   = se.fill()
    q_report = se.quality_report()

    # Multi-dataset gap transfer
    datasets_info = [...]   # see transfer module for schema
    matches = find_matching_gaps(datasets_info)
    transfers = transfer_samples(matches)
"""

from .core      import SynthEdge
from .quality   import classify_severity, gap_region_kl
from .scanner   import scan, adaptive_bins
from .transfer  import find_matching_gaps, transfer_samples, apply_transfers
from .report    import generate_report

__version__ = "0.1.0"
__all__ = [
    "SynthEdge",
    "classify_severity",
    "gap_region_kl",
    "scan",
    "adaptive_bins",
    "find_matching_gaps",
    "transfer_samples",
    "apply_transfers",
    "generate_report",
]
