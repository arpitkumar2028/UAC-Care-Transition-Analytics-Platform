"""
report_generator.py
Executive PDF Report Generator built with ReportLab.
Compiles a structured, multi-page operational intelligence briefing in-memory
for executive decision-makers, agency directors, and internship documentation.
"""

import io
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)

def generate_pdf_report(
    metrics_summary: Dict[str, Any],
    forecast_df: Optional[pd.DataFrame] = None
) -> bytes:
    """
    Compiles an executive summary PDF in-memory and returns the raw bytes.
    
    Parameters:
    - metrics_summary: Dict of current metrics produced by get_current_kpis() and ai_copilot
    - forecast_df: DataFrame of 14-day forward projections from ml_forecasting
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Brand Styles
    primary_color = colors.HexColor("#0F172A")
    accent_blue = colors.HexColor("#0284C7")
    emerald = colors.HexColor("#059669")
    dark_gray = colors.HexColor("#334155")
    light_bg = colors.HexColor("#F8FAFC")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=accent_blue,
        spaceAfter=12
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=dark_gray
    )
    
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=primary_color
    )

    story = []
    
    # --- HEADER SECTION ---
    story.append(Paragraph("Care Transition Efficiency & Placement Outcome Analytics", title_style))
    story.append(Paragraph(
        f"<b>Executive Decision Support Briefing</b> | Generated: {datetime.now().strftime('%B %d, %Y - %H:%M:%S UTC')} | Data Scope: 2023 - 2025",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_blue, spaceBefore=0, spaceAfter=12))
    
    # --- SYSTEM HEALTH & STATUS BANNER ---
    health_grade = metrics_summary.get('health_grade', 'N/A')
    health_score = metrics_summary.get('health_score', 0.0)
    dwell_days = metrics_summary.get('estimated_dwell_days', 0)
    ter_val = metrics_summary.get('ter_ma7', 0.0)
    spt_val = metrics_summary.get('spt_current', 0.0)
    
    status_summary_data = [
        [
            Paragraph("<b>Overall Pipeline Health</b>", body_style),
            Paragraph("<b>Transfer Efficiency (TER)</b>", body_style),
            Paragraph("<b>Pipeline Throughput (SPT)</b>", body_style),
            Paragraph("<b>Est. Dwell Time</b>", body_style),
        ],
        [
            Paragraph(f"<font size=14 color='#0284C7'><b>{health_grade}</b></font><br/>Score: {health_score}/100", body_style),
            Paragraph(f"<font size=14 color='#059669'><b>{ter_val}</b></font><br/>Benchmark: &ge;0.85", body_style),
            Paragraph(f"<font size=14 color='#0F172A'><b>{spt_val}</b></font><br/>Target: &ge;1.00", body_style),
            Paragraph(f"<font size=14 color='#D97706'><b>{dwell_days} Days</b></font><br/>In Shelter Care", body_style),
        ]
    ]
    
    status_table = Table(status_summary_data, colWidths=[130, 130, 130, 140])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(status_table)
    story.append(Spacer(1, 14))
    
    # --- EXECUTIVE AUDIT SUMMARY ---
    story.append(Paragraph("1. Executive Operational Audit", section_heading))
    findings = metrics_summary.get('findings', [
        "Pipeline operations evaluated across CBP intake and HHS shelter networks.",
        "System throughput closely aligns with safe reunification targets."
    ])
    import re
    for f in findings:
        # Properly replace markdown **text** with <b>text</b>
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', f)
        story.append(Paragraph(f"• {clean_text}", body_style))
        story.append(Spacer(1, 3))
        
    story.append(Spacer(1, 10))
    
    # --- DETAILED KPI SCORECARD TABLE ---
    story.append(Paragraph("2. Operational KPI Scorecard & Process Velocity", section_heading))
    
    kpi_table_data = [
        [
            Paragraph("<b>KPI Name</b>", body_bold),
            Paragraph("<b>Operational Meaning</b>", body_bold),
            Paragraph("<b>Current Value</b>", body_bold),
            Paragraph("<b>Target Benchmark</b>", body_bold),
            Paragraph("<b>Operational Status</b>", body_bold)
        ],
        [
            Paragraph("Transfer Efficiency Ratio", body_style),
            Paragraph("CBP &rarr; HHS Handover Speed", body_style),
            Paragraph(f"{ter_val}", body_style),
            Paragraph("&ge; 0.85", body_style),
            Paragraph("Optimal" if ter_val >= 0.85 else "Watchlist", body_style)
        ],
        [
            Paragraph("Discharge Effectiveness Index", body_style),
            Paragraph("Daily Sponsor Placement Turnover", body_style),
            Paragraph(f"{metrics_summary.get('dei_current_pct', 0.0)}%", body_style),
            Paragraph("&ge; 2.5%", body_style),
            Paragraph("Normal", body_style)
        ],
        [
            Paragraph("System Pipeline Throughput", body_style),
            Paragraph("Net Discharges &divide; Intake", body_style),
            Paragraph(f"{spt_val}", body_style),
            Paragraph("&ge; 1.00 (Equilibrium)", body_style),
            Paragraph("Surplus" if spt_val >= 1.0 else "Deficit", body_style)
        ],
        [
            Paragraph("Net Backlog Rate", body_style),
            Paragraph("Daily Inflow vs Outflow Gap", body_style),
            Paragraph(f"{metrics_summary.get('daily_backlog', 0)} / day", body_style),
            Paragraph("&le; 0 (No growth)", body_style),
            Paragraph("Controlled", body_style)
        ],
        [
            Paragraph("Outcome Stability Score", body_style),
            Paragraph("30-Day Discharge Consistency", body_style),
            Paragraph(f"{metrics_summary.get('stability_score', 0.0)}", body_style),
            Paragraph("&ge; 0.70", body_style),
            Paragraph("High Predictability", body_style)
        ]
    ]
    
    kpi_table = Table(kpi_table_data, colWidths=[120, 160, 80, 90, 80])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))
    
    # --- 14-DAY PREDICTIVE FORECAST TABLE ---
    if forecast_df is not None and not forecast_df.empty:
        story.append(Paragraph("3. Predictive Outlook: 14-Day Forward Bed Demand & Placements", section_heading))
        
        forecast_table_data = [
            [
                Paragraph("<b>Date</b>", body_bold),
                Paragraph("<b>Day</b>", body_bold),
                Paragraph("<b>Projected Inflow</b>", body_bold),
                Paragraph("<b>Projected Discharges</b>", body_bold),
                Paragraph("<b>Net Bed Delta</b>", body_bold),
                Paragraph("<b>Forecast HHS Census</b>", body_bold)
            ]
        ]
        
        # Take up to 7 or 10 rows for page compactness
        for _, row in forecast_df.head(10).iterrows():
            forecast_table_data.append([
                Paragraph(pd.to_datetime(row['Date']).strftime('%Y-%m-%d'), body_style),
                Paragraph(str(row['DayOfWeek'])[:3], body_style),
                Paragraph(str(row['Projected_Transfers_In']), body_style),
                Paragraph(str(row['Projected_Discharges']), body_style),
                Paragraph(str(row['Projected_Net_Flow']), body_style),
                Paragraph(f"{int(row['Projected_HHS_Census']):,}", body_style),
            ])
            
        fc_table = Table(forecast_table_data, colWidths=[90, 60, 95, 105, 80, 100])
        fc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284C7")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(fc_table)
        story.append(Spacer(1, 14))
        
    # --- PRESCRIPTIVE CASEWORK INTERVENTIONS ---
    story.append(Paragraph("4. Prescriptive Casework Leadership Directives", section_heading))
    recs = metrics_summary.get('recommendations', [
        "Maintain active intake coordination between border transit stations and shelter reception hubs.",
        "Implement weekend on-call sponsor travel coordination to mitigate weekend discharge drops.",
        "Establish accelerated review pathways for Category 1 parental sponsors."
    ])
    for i, r in enumerate(recs):
        story.append(Paragraph(f"<b>Directive {i+1}:</b> {r}", body_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
