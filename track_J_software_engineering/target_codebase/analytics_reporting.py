"""
Analytics & Financial Reporting Engine:
Performs ABC inventory categorization (Pareto analysis), gross margin calculations,
turnover velocity, customer lifetime value and churn risk estimation.
"""

from __future__ import annotations
import math
from datetime import date, timedelta
from typing import Dict, List, Tuple, Set, Optional

from track_J_software_engineering.target_codebase.models import (
    Item, ItemCategory, Customer, CustomerTier, Order, Invoice, OrderStatus
)
from track_J_software_engineering.target_codebase.inventory_service import InventoryService


class AnalyticsReportingEngine:
    def __init__(self, inventory_service: InventoryService):
        self.inventory_service = inventory_service

    def compute_inventory_valuation(self) -> Dict[str, float]:
        """
        Calculates total inventory asset valuation grouped by product category.
        """
        valuation_by_category: Dict[str, float] = {}
        total_valuation = 0.0

        for sku, item in self.inventory_service._items.items():
            on_hand = self.inventory_service.get_on_hand(sku)
            asset_value = on_hand * item.unit_cost
            cat_name = item.category.name
            valuation_by_category[cat_name] = valuation_by_category.get(cat_name, 0.0) + asset_value
            total_valuation += asset_value

        valuation_by_category["TOTAL"] = total_valuation
        return valuation_by_category

    def perform_abc_analysis(self, annual_sales_units: Dict[str, int]) -> Dict[str, str]:
        """
        Performs Pareto 80/20 ABC inventory classification:
        - Category A: Top 80% of cumulative monetary value (tight control)
        - Category B: Next 15% (80% to 95%) of cumulative monetary value
        - Category C: Final 5% of cumulative monetary value
        """
        sku_values: List[Tuple[str, float]] = []
        for sku, units in annual_sales_units.items():
            item = self.inventory_service._items.get(sku)
            if item:
                val = units * item.base_price
                sku_values.append((sku, val))

        sku_values.sort(key=lambda x: x[1], reverse=True)
        total_revenue = sum(val for _, val in sku_values)
        if total_revenue <= 0:
            return {sku: "C" for sku, _ in sku_values}

        classification: Dict[str, str] = {}
        cum_revenue = 0.0

        for sku, val in sku_values:
            prior_pct = (cum_revenue / total_revenue) * 100.0
            if prior_pct < 80.0:
                classification[sku] = "A"
            elif prior_pct < 95.0:
                classification[sku] = "B"
            else:
                classification[sku] = "C"
            cum_revenue += val

        return classification

    def calculate_inventory_turnover(
        self,
        cogs_annual: float,
        beginning_inventory_val: float,
        ending_inventory_val: float
    ) -> float:
        """
        Inventory Turnover Ratio = Cost of Goods Sold (COGS) / Average Inventory Value
        """
        avg_inv = (beginning_inventory_val + ending_inventory_val) / 2.0
        if avg_inv <= 0:
            return 0.0
        return round(cogs_annual / avg_inv, 2)

    def identify_dead_stock(
        self,
        threshold_days: int = 90,
        current_date: Optional[date] = None
    ) -> List[Dict[str, object]]:
        """
        Identifies batches sitting in storage without movement exceeding threshold_days.
        """
        today = current_date or date.today()
        dead_stock_records = []

        for sku, batches in self.inventory_service._batches.items():
            item = self.inventory_service._items.get(sku)
            if not item:
                continue
            for batch in batches:
                age_days = (today - batch.received_date).days
                if age_days >= threshold_days and batch.quantity > 0:
                    capital_tied_up = batch.quantity * batch.unit_cost
                    dead_stock_records.append({
                        "sku": sku,
                        "batch_id": batch.batch_id,
                        "warehouse_id": batch.warehouse_id,
                        "quantity": batch.quantity,
                        "age_days": age_days,
                        "tied_up_capital": round(capital_tied_up, 2)
                    })

        return dead_stock_records

    def summarize_sales_by_tier(
        self,
        invoices: List[Invoice],
        customers: Dict[str, Customer]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregates financial performance metrics broken down by customer relationship tier.
        """
        summary = {
            tier.name: {"total_revenue": 0.0, "total_tax": 0.0, "invoice_count": 0}
            for tier in CustomerTier
        }

        for inv in invoices:
            cust = customers.get(inv.customer_id)
            if not cust:
                continue
            tier_name = cust.tier.name
            summary[tier_name]["total_revenue"] += inv.subtotal
            summary[tier_name]["total_tax"] += inv.tax_amount
            summary[tier_name]["invoice_count"] += 1

        return summary
