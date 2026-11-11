"""
ai_copilot.py
Local Operational AI Copilot for UAC Care Transition Efficiency.
Runs 100% locally with zero paid API keys, providing rule-backed diagnostic audits,
bottleneck identification, and operational decision support.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .metrics_engine import compute_pipeline_metrics, get_current_kpis

class PipelineCopilot:
    """
    Expert rule-backed operational intelligence copilot.
    Analyzes custody queues, transition efficiency ratios, and discharge outcomes
    to diagnose pipeline friction and prescribe caseworker interventions.
    """
    
    def __init__(self, df: pd.DataFrame):
        # Ensure metrics are calculated
        if 'Transfer_Efficiency_Ratio' not in df.columns:
            self.df = compute_pipeline_metrics(df)
        else:
            self.df = df.copy()
            
        self.kpis = get_current_kpis(self.df)
        self._analyze_pipeline_dynamics()
        
    def _analyze_pipeline_dynamics(self):
        """Precomputes system diagnostics and behavioral markers."""
        recent_30 = self.df.iloc[-30:] if len(self.df) >= 30 else self.df
        
        self.recent_ter = recent_30['Transfer_Efficiency_Ratio'].mean()
        self.recent_spt = recent_30['Pipeline_Throughput_MA7'].mean()
        self.recent_dei = recent_30['Discharge_Effectiveness_Pct'].mean()
        self.recent_apprehensions = recent_30['Apprehensions'].mean()
        self.recent_discharges = recent_30['Discharges_HHS'].mean()
        self.recent_cbp_custody = recent_30['CBP_Custody'].mean()
        self.recent_hhs_care = recent_30['HHS_Care'].mean()
        
        # Primary Bottleneck Attribution
        # If TER is low (< 0.6) and CBP custody is elevated (> 150), Stage 1 CBP transition is the primary choke point
        # If DEI is low (< 2.5%) and HHS Care is high, Stage 2 Sponsor vetting is the primary choke point
        if self.recent_ter < 0.60 and self.recent_cbp_custody > 100:
            self.primary_bottleneck = "Stage 1 (Border Patrol $\\rightarrow$ HHS Transfer Friction)"
            self.bottleneck_stage = 1
        elif self.recent_dei < 2.5 and self.recent_hhs_care > 5000:
            self.primary_bottleneck = "Stage 2 (In-Care Case Management & Sponsor Vetting Latency)"
            self.bottleneck_stage = 2
        elif self.recent_spt < 0.85:
            self.primary_bottleneck = "System-Wide Backlog Accumulation (Intake Outpacing Discharges)"
            self.bottleneck_stage = 0
        else:
            self.primary_bottleneck = "None (Pipeline Operating in Dynamic Equilibrium)"
            self.bottleneck_stage = -1
            
        # Weekend Analysis
        wkday_discharges = self.df[self.df['IsWeekend'] == 0]['Discharges_HHS'].mean()
        wkend_discharges = self.df[self.df['IsWeekend'] == 1]['Discharges_HHS'].mean()
        self.weekend_discharge_drop_pct = (
            ((wkday_discharges - wkend_discharges) / wkday_discharges * 100)
            if wkday_discharges > 0 else 0
        )
        
    def generate_audit_summary(self) -> Dict[str, Any]:
        """
        Generates an end-to-end executive audit summary of pipeline efficiency.
        """
        health_grade = self.kpis['health_grade']
        health_score = self.kpis['health_score']
        
        # Determine status highlights
        findings = []
        if self.recent_ter < 0.50:
            findings.append(f"⚠️ **CBP Transfer Latency Alert**: Transfer Efficiency Ratio averages {self.recent_ter:.2f}, indicating significant border facility queue buildup before HHS intake.")
        else:
            findings.append(f"✅ **CBP Transfer Clearance**: Transfer Efficiency Ratio is healthy at {self.recent_ter:.2f}, clearing border arrivals promptly.")
            
        if self.recent_spt < 0.90:
            findings.append(f"⚠️ **Backlog Accumulation Alert**: System Throughput is {self.recent_spt:.2f}. Discharges ({self.recent_discharges:.0f}/day) are failing to match inflow ({self.recent_apprehensions:.0f}/day), causing a net case deficit.")
        else:
            findings.append(f"✅ **System Equilibrium**: Pipeline Throughput is {self.recent_spt:.2f}, successfully discharging children at or above arrival velocity.")
            
        if self.weekend_discharge_drop_pct > 30:
            findings.append(f"📅 **Weekend Stagnation Marker**: Weekend discharges drop by **{self.weekend_discharge_drop_pct:.1f}%** relative to weekdays due to administrative and background-check downtime.")
            
        # Strategic recommendations
        recommendations = []
        if self.bottleneck_stage == 1:
            recommendations.append("Mobilize emergency bed intake coordinators at primary ORR transit hubs to expedite CBP handoffs within the 72-hour window.")
        elif self.bottleneck_stage == 2:
            recommendations.append("Prioritize Category 1 (parent/legal guardian) sponsor vetting fast-tracking to reduce shelter dwell time below 30 days.")
        else:
            recommendations.append("Maintain active roster of standby foster care and transitional shelter beds for potential border arrival surges.")
            
        recommendations.append("Implement weekend on-call sponsor travel coordination to mitigate the 30%+ weekend discharge slowdown.")
        
        briefing_markdown = f"""
### 📋 Executive Care Transition Audit
* **As of Date**: {self.kpis['as_of_date']}
* **System Health Grade**: **{health_grade}** ({health_score}/100)
* **Primary Operational Bottleneck**: **{self.primary_bottleneck}**
* **Estimated Dwell Time**: **{self.kpis['estimated_dwell_days']} days** in shelter care

#### Key Operational Findings:
""" + "\n".join([f"- {f}" for f in findings]) + f"""

#### Prescriptive Casework Recommendations:
""" + "\n".join([f"1. {r}" for r in recommendations])

        return {
            "health_grade": health_grade,
            "health_score": health_score,
            "primary_bottleneck": self.primary_bottleneck,
            "estimated_dwell_days": self.kpis['estimated_dwell_days'],
            "recent_ter": round(self.recent_ter, 2),
            "recent_spt": round(self.recent_spt, 2),
            "weekend_drop_pct": round(self.weekend_discharge_drop_pct, 1),
            "findings": findings,
            "recommendations": recommendations,
            "briefing_markdown": briefing_markdown
        }
        
    def ask_copilot(self, query: str) -> str:
        """
        Rule-backed NLP response system answering caseworker and leadership questions
        directly grounded on live dataset numbers.
        """
        q = query.lower().strip()
        k = self.kpis
        
        # 1. Bottleneck & Friction
        if any(w in q for w in ['bottleneck', 'choke', 'delay', 'where', 'slowdown', 'friction']):
            return (
                f"**Diagnostic Finding**: The primary bottleneck is currently **{self.primary_bottleneck}**.\n\n"
                f"• **Stage 1 Status (CBP $\\rightarrow$ HHS)**: TER is **{k['ter_ma7']}** with **{k['cbp_custody']}** children in CBP custody.\n"
                f"• **Stage 2 Status (HHS $\\rightarrow$ Sponsor)**: DEI is **{k['dei_current_pct']}%** with **{k['hhs_care']:,}** children in shelter beds.\n"
                f"• **Net Flow Balance**: Intake is averaging **{self.recent_apprehensions:.0f}/day** while discharges are **{self.recent_discharges:.0f}/day**."
            )
            
        # 2. Transfer Efficiency Ratio (TER)
        if any(w in q for w in ['transfer efficiency', 'ter', 'cbp to hhs', 'cbp custody', 'transfer speed']):
            eval_text = "healthy (> 0.85)" if k['ter_ma7'] >= 0.85 else "suboptimal (< 0.85) indicating transfer latency"
            return (
                f"**Transfer Efficiency Ratio (TER)** measures how quickly children are transferred out of CBP holding into HHS care relative to total active CBP custody.\n\n"
                f"• **Current 7-Day TER**: **{k['ter_ma7']}** (Status: {eval_text}).\n"
                f"• **30-Day Trend**: Changed by **{k['ter_delta_pct']}%** compared to 30 days prior.\n"
                f"• **Regulatory Context**: Under TVPRA, children must be transferred within 72 hours. A TER below 0.5 indicates risk of exceeding this statutory window."
            )
            
        # 3. System Pipeline Throughput (SPT) & Backlog
        if any(w in q for w in ['throughput', 'spt', 'backlog', 'keep pace', 'inflow', 'deficit', 'accumulation']):
            balance_text = "clearing backlogged cases" if k['spt_current'] >= 1.0 else "accumulating active cases in shelter custody"
            return (
                f"**Pipeline Throughput (SPT)** measures total sponsor discharges divided by total border apprehensions.\n\n"
                f"• **Current Throughput Rate**: **{k['spt_current']}**.\n"
                f"• **Operational Meaning**: The system is currently **{balance_text}**.\n"
                f"• **Daily Net Balance**: **{k['daily_backlog']} children/day**.\n"
                f"• **Cumulative System Backlog**: **{k['cumulative_backlog']:,} cases** accumulated across the reporting span."
            )
            
        # 4. Weekend Delay Analysis
        if any(w in q for w in ['weekend', 'saturday', 'sunday', 'day of week', 'weekday']):
            return (
                f"**Weekend Latency Analysis**:\n\n"
                f"• **Discharge Velocity Drop**: Weekend sponsor discharges drop by **{self.weekend_discharge_drop_pct:.1f}%** compared to weekdays.\n"
                f"• **Root Cause**: Court closures, notary/background check delays, and reduced airline/travel escort availability on Saturdays and Sundays.\n"
                f"• **Impact**: Leads to recurring 'Monday Surges' where CBP queues spike due to uninterrupted border intake during weekend case processing lulls."
            )
            
        # 5. Dwell Time / Length of Stay
        if any(w in q for w in ['dwell time', 'length of stay', 'los', 'days in care', 'how long']):
            return (
                f"**Shelter Dwell Time Analysis (Little's Law Proxy)**:\n\n"
                f"• **Estimated Average Dwell Time**: **{k['estimated_dwell_days']} days**.\n"
                f"• **30-Day Shift**: Shifted by **{k['dwell_delta']} days**.\n"
                f"• **Target Benchmark**: Best practice child welfare standards aim for sponsor reunification within **30 to 35 days** of initial intake."
            )
            
        # 6. Recommendations & Action Plan
        if any(w in q for w in ['recommend', 'action', 'improve', 'caseworker', 'fix', 'strategy']):
            summary = self.generate_audit_summary()
            recs = "\n".join([f"{i+1}. {r}" for i, r in enumerate(summary['recommendations'])])
            return f"**Targeted Action Plan for Casework Leadership**:\n\n{recs}"
            
        # Default Overview
        return self.generate_audit_summary()['briefing_markdown']
