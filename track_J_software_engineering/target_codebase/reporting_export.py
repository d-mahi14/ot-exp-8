"""
Reporting and Export Module: Data export, CSV generation, and reconciliation reports.
"""

from __future__ import annotations
import csv
import io
from typing import List, Dict, Any, Optional

from track_J_software_engineering.target_codebase.models import Item, Invoice, Order


class DataExportService:
    """Provides tabular CSV serialization and reconciliation exports."""

    @staticmethod
    def export_items_to_csv(items: List[Item]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["SKU", "Name", "Category", "UnitCost", "BasePrice", "GrossMarginPct", "LeadTimeDays"])
        for it in items:
            writer.writerow([
                it.sku,
                it.name,
                it.category.name,
                f"{it.unit_cost:.2f}",
                f"{it.base_price:.2f}",
                f"{it.gross_margin * 100.0:.2f}%",
                it.lead_time_days
            ])
        return output.getvalue()

    @staticmethod
    def export_invoices_to_csv(invoices: List[Invoice]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["InvoiceID", "OrderID", "CustomerID", "Subtotal", "Tax", "Shipping", "Total", "Paid"])
        for inv in invoices:
            writer.writerow([
                inv.invoice_id,
                inv.order_id,
                inv.customer_id,
                f"{inv.subtotal:.2f}",
                f"{inv.tax_amount:.2f}",
                f"{inv.shipping_fee:.2f}",
                f"{inv.total_amount:.2f}",
                inv.paid
            ])
        return output.getvalue()

    @staticmethod
    def generate_audit_summary_report(audit_trail: List[Dict[str, str]]) -> Dict[str, int]:
        event_counts: Dict[str, int] = {}
        for entry in audit_trail:
            event = entry.get("event", "UNKNOWN")
            event_counts[event] = event_counts.get(event, 0) + 1
        return event_counts
