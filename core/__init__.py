"""
core package for quantitative analytics and process metrics.
"""
from core.analytics import (
    compute_transfer_efficiency_ratio,
    compute_discharge_effectiveness_index,
    compute_pipeline_throughput_rate,
    compute_backlog_accumulation_rate,
    detect_backlog_anomalies,
    compute_full_analytics_suite,
    normalize_pipeline_dataframe
)

__all__ = [
    "compute_transfer_efficiency_ratio",
    "compute_discharge_effectiveness_index",
    "compute_pipeline_throughput_rate",
    "compute_backlog_accumulation_rate",
    "detect_backlog_anomalies",
    "compute_full_analytics_suite",
    "normalize_pipeline_dataframe"
]

