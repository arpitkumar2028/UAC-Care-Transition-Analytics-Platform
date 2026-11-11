"""
db/models.py
Data models and typed schemas for the UAC Care Transition Efficiency platform.
Provides typed dataclasses and dynamic serialization for database operations.
"""

from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional, Dict, Any, List
import uuid

@dataclass
class DailyPipelineMetric:
    """
    Macro time-series metrics capturing federal custody flow & stocks.
    """
    date: str  # ISO-8601 YYYY-MM-DD
    cbp_apprehensions: int
    cbp_active_custody: int
    cbp_transfers_out: int
    hhs_active_care: int
    hhs_discharges: int
    sponsor_vetted_reunifications: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Any) -> "DailyPipelineMetric":
        return cls(
            date=str(row["date"]),
            cbp_apprehensions=int(row["cbp_apprehensions"]),
            cbp_active_custody=int(row["cbp_active_custody"]),
            cbp_transfers_out=int(row["cbp_transfers_out"]),
            hhs_active_care=int(row["hhs_active_care"]),
            hhs_discharges=int(row["hhs_discharges"]),
            sponsor_vetted_reunifications=int(row["sponsor_vetted_reunifications"])
        )

@dataclass
class CaseLevelEvent:
    """
    Micro event ledger tracking individual child transitions through the pipeline.
    """
    case_id: str
    apprehension_date: str
    processing_facility_id: str
    sponsor_risk_tier: str  # 'Low', 'Medium', 'High'
    status: str             # 'CBP_Custody', 'HHS_Care', 'Discharged', 'Escalated'
    cbp_transfer_date: Optional[str] = None
    hhs_intake_date: Optional[str] = None
    discharge_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Any) -> "CaseLevelEvent":
        return cls(
            case_id=str(row["case_id"]),
            apprehension_date=str(row["apprehension_date"]),
            cbp_transfer_date=str(row["cbp_transfer_date"]) if row["cbp_transfer_date"] else None,
            hhs_intake_date=str(row["hhs_intake_date"]) if row["hhs_intake_date"] else None,
            discharge_date=str(row["discharge_date"]) if row["discharge_date"] else None,
            processing_facility_id=str(row["processing_facility_id"]),
            sponsor_risk_tier=str(row["sponsor_risk_tier"]),
            status=str(row["status"])
        )

