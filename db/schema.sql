-- db/schema.sql
-- Production DDL for UAC Care Transition Efficiency & Placement Outcome Analytics
-- Compatible with SQLite and PostgreSQL

-- Table 1: Daily Pipeline Macro Metrics
CREATE TABLE IF NOT EXISTS daily_pipeline_metrics (
    date DATE PRIMARY KEY,
    cbp_apprehensions INTEGER NOT NULL CHECK (cbp_apprehensions >= 0),
    cbp_active_custody INTEGER NOT NULL CHECK (cbp_active_custody >= 0),
    cbp_transfers_out INTEGER NOT NULL CHECK (cbp_transfers_out >= 0),
    hhs_active_care INTEGER NOT NULL CHECK (hhs_active_care >= 0),
    hhs_discharges INTEGER NOT NULL CHECK (hhs_discharges >= 0),
    sponsor_vetted_reunifications INTEGER NOT NULL CHECK (sponsor_vetted_reunifications >= 0)
);

CREATE INDEX IF NOT EXISTS idx_daily_pipeline_date ON daily_pipeline_metrics (date DESC);

-- Table 2: Case Level Micro Events
CREATE TABLE IF NOT EXISTS case_level_events (
    case_id VARCHAR(36) PRIMARY KEY,
    apprehension_date DATE NOT NULL,
    cbp_transfer_date DATE,
    hhs_intake_date DATE,
    discharge_date DATE,
    processing_facility_id VARCHAR(30) NOT NULL,
    sponsor_risk_tier VARCHAR(10) NOT NULL CHECK (sponsor_risk_tier IN ('Low', 'Medium', 'High')),
    status VARCHAR(20) NOT NULL CHECK (status IN ('CBP_Custody', 'HHS_Care', 'Discharged', 'Escalated'))
);

CREATE INDEX IF NOT EXISTS idx_case_apprehension_date ON case_level_events (apprehension_date);
CREATE INDEX IF NOT EXISTS idx_case_discharge_date ON case_level_events (discharge_date);
CREATE INDEX IF NOT EXISTS idx_case_status ON case_level_events (status);
CREATE INDEX IF NOT EXISTS idx_case_risk_tier ON case_level_events (sponsor_risk_tier);
CREATE INDEX IF NOT EXISTS idx_case_facility ON case_level_events (processing_facility_id);

