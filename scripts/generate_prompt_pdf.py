#!/usr/bin/env python3
"""
generate_prompt_pdf.py
Compiles all Markdown prompt records in prompts/ into AI_Prompt_Log.pdf.
Uses ReportLab to generate a clean, publication-ready submission artifact.
"""

import os
import re
import glob
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and display 'Page X of Y'."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#555555"))
        # Header rule
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.5)
        self.line(40, letter[1] - 35, letter[0] - 40, letter[1] - 35)
        self.drawString(40, letter[1] - 30, "Optimization Techniques — Assignment 8: AI Prompt & Vibe-Coding Audit Log")

        # Footer rule
        self.line(40, 42, letter[0] - 40, 42)
        self.drawString(40, 28, f"Confidential & Academic Record • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 40, 28, page_str)
        self.restoreState()


def parse_prompt_file(filepath: Path) -> dict:
    """Parses a markdown prompt file into structured fields."""
    content = filepath.read_text(encoding="utf-8")

    data = {
        "title": filepath.stem.replace("_", " ").title(),
        "prompt_id": "P-UNKNOWN",
        "date": "Unknown Date",
        "track": "General",
        "ai_tool": "Antigravity",
        "commit": "N/A",
        "purpose": "",
        "original_prompt": "",
        "ai_response": "",
        "files_changed": "",
        "verification": "",
        "human_revisions": "",
        "raw": content
    }

    # Extract metadata fields
    m_id = re.search(r"\*\*Prompt ID\*\*:\s*`?([^`\n\r]+)`?", content)
    if m_id: data["prompt_id"] = m_id.group(1).strip()

    m_date = re.search(r"\*\*Date & Time\*\*:\s*([^\n\r]+)", content)
    if m_date: data["date"] = m_date.group(1).strip()

    m_track = re.search(r"\*\*Track\*\*:\s*([^\n\r]+)", content)
    if m_track: data["track"] = m_track.group(1).strip()

    m_tool = re.search(r"\*\*AI Tool\*\*:\s*([^\n\r]+)", content)
    if m_tool: data["ai_tool"] = m_tool.group(1).strip()

    m_commit = re.search(r"\*\*Commit Reference\*\*:\s*([^\n\r]+)", content)
    if m_commit: data["commit"] = m_commit.group(1).strip()

    # Extract sections
    def extract_section(section_name: str) -> str:
        pattern = rf"###\s+{re.escape(section_name)}[\r\n]+(.*?)(?=###|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    data["purpose"] = extract_section("Purpose")
    raw_prompt = extract_section("Original Prompt")
    # Clean code blocks
    code_match = re.search(r"```(?:text)?[\r\n]+(.*?)```", raw_prompt, re.DOTALL)
    if code_match:
        data["original_prompt"] = code_match.group(1).strip()
    else:
        data["original_prompt"] = raw_prompt

    data["ai_response"] = extract_section("AI Response Summary")
    data["files_changed"] = extract_section("Files Created / Modified")
    data["verification"] = extract_section("Verification Note")
    data["human_revisions"] = extract_section("Human Revisions / Instructions Received")

    return data


def build_pdf(output_pdf_path: str = "AI_Prompt_Log.pdf"):
    """Compiles all prompt logs in prompts/ into a unified PDF."""
    prompts_dir = Path("prompts")
    if not prompts_dir.exists():
        print(f"Error: Directory '{prompts_dir}' does not exist.")
        return

    # Collect all markdown files in structured order
    categories = [
        "general", "track_A", "track_B", "track_C", "track_D",
        "track_E", "track_F", "track_G", "track_H", "track_I", "track_J"
    ]
    md_files = []
    for cat in categories:
        cat_dir = prompts_dir / cat
        if cat_dir.exists():
            files = sorted(cat_dir.glob("*.md"))
            md_files.extend(files)

    if not md_files:
        print("No markdown prompt files found.")
        return

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0969DA"),
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#24292F"),
        spaceAfter=20
    )

    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#1F2328"),
        spaceBefore=14,
        spaceAfter=6
    )

    h3_style = ParagraphStyle(
        "H3Style",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0969DA"),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1F2328")
    )

    code_style = ParagraphStyle(
        "CodeBoxStyle",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#0A3069"),
        backColor=colors.HexColor("#F6F8FA")
    )

    story = []

    # Title Banner
    story.append(Paragraph("AI Prompt & Vibe-Coding Audit Log", title_style))
    story.append(Paragraph(
        "<b>Course:</b> Optimization Techniques — Assignment 8 (The 14-Day Vibe-Coding Challenge)<br/>"
        f"<b>Total Logged Interactions:</b> {len(md_files)} | <b>Report Compiled:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        "<b>Strict Vibe-Coding Disclosure:</b> Complete chronological trail of human prompts, AI responses, code changes, and verification outcomes.",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0969DA"), spaceAfter=15))

    # Summary Index Table
    story.append(Paragraph("Prompt Index", h2_style))
    summary_data = [["ID", "Date", "Track", "Purpose", "Model"]]
    for f in md_files:
        d = parse_prompt_file(f)
        purpose_snippet = (d["purpose"][:55] + "...") if len(d["purpose"]) > 55 else d["purpose"]
        summary_data.append([
            d["prompt_id"],
            d["date"][:10],
            d["track"][:12],
            purpose_snippet or d["title"],
            d["ai_tool"][:18]
        ])

    idx_table = Table(summary_data, colWidths=[65, 70, 80, 240, 77])
    idx_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0969DA")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")])
    ]))
    story.append(idx_table)
    story.append(Spacer(1, 15))
    story.append(PageBreak())

    # Detailed entries
    for idx, f in enumerate(md_files):
        d = parse_prompt_file(f)
        
        story.append(Paragraph(f"Interaction #{idx + 1}: {d['prompt_id']} — {d['title']}", h2_style))
        
        # Meta table
        meta = [
            [Paragraph("<b>Date & Time:</b>", body_style), Paragraph(d["date"], body_style),
             Paragraph("<b>Track:</b>", body_style), Paragraph(d["track"], body_style)],
            [Paragraph("<b>AI Tool/Model:</b>", body_style), Paragraph(d["ai_tool"], body_style),
             Paragraph("<b>Commit Ref:</b>", body_style), Paragraph(d["commit"], body_style)]
        ]
        meta_table = Table(meta, colWidths=[80, 185, 80, 187])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEEF2")),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 6))

        # Purpose
        if d["purpose"]:
            story.append(Paragraph("Purpose", h3_style))
            story.append(Paragraph(d["purpose"].replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 4))

        # Original Prompt
        story.append(Paragraph("Original Prompt (Verbatim)", h3_style))
        # Truncate visually for display if prompt is massive, but keep up to 2500 chars
        prompt_text = d["original_prompt"]
        if len(prompt_text) > 3000:
            prompt_text = prompt_text[:3000] + "\n... [Truncated for PDF page layout — full text preserved in markdown log] ..."
        
        prompt_box = Table([[Preformatted(prompt_text, code_style)]], colWidths=[532])
        prompt_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#AFB8C1")),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(prompt_box)
        story.append(Spacer(1, 6))

        # AI Response Summary
        if d["ai_response"]:
            story.append(Paragraph("AI Response Summary", h3_style))
            clean_resp = d["ai_response"].replace("\n", "<br/>")
            clean_resp = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", clean_resp)
            story.append(Paragraph(clean_resp, body_style))
            story.append(Spacer(1, 4))

        # Files and verification in 2-column or list
        if d["files_changed"] or d["verification"]:
            fv_data = []
            if d["files_changed"]:
                fv_data.append([Paragraph("<b>Files Modified / Created:</b>", body_style),
                                Paragraph(d["files_changed"].replace("\n", "<br/>"), body_style)])
            if d["verification"]:
                fv_data.append([Paragraph("<b>Independent Verification:</b>", body_style),
                                Paragraph(d["verification"].replace("\n", "<br/>"), body_style)])
            if d["human_revisions"]:
                fv_data.append([Paragraph("<b>Human Guidance / Revisions:</b>", body_style),
                                Paragraph(d["human_revisions"].replace("\n", "<br/>"), body_style)])
            
            fv_table = Table(fv_data, colWidths=[150, 382])
            fv_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FBFCFD")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEEF2")),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(fv_table)

        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#D0D7DE"), spaceAfter=10))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated prompt PDF at: {output_pdf_path}")


if __name__ == "__main__":
    build_pdf("AI_Prompt_Log.pdf")
