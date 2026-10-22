"""
Pricing & Tax Engine: Multi-tier dynamic discounting, coupon verification,
regional sales tax policies, multi-currency conversion, and weight-bracket shipping logic.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from track_J_software_engineering.target_codebase.models import (
    Customer, CustomerTier, Order, OrderLine, Item
)


@dataclass
class Coupon:
    code: str
    discount_pct: float
    min_order_amount: float = 0.0
    expiry_date: Optional[date] = None
    max_discount_amount: float = 500.0
    active: bool = True

    def is_valid(self, order_amount: float, current_date: Optional[date] = None) -> bool:
        if not self.active:
            return False
        if order_amount < self.min_order_amount:
            return False
        if self.expiry_date:
            now = current_date or date.today()
            if self.expiry_date < now:
                return False
        return True

    def compute_discount(self, order_amount: float) -> float:
        raw_disc = order_amount * (self.discount_pct / 100.0)
        return min(raw_disc, self.max_discount_amount)


class PricingEngine:
    def __init__(self):
        self._coupons: Dict[str, Coupon] = {}
        # Exchange rates relative to USD base
        self._exchange_rates: Dict[str, float] = {
            "USD": 1.0,
            "EUR": 0.92,
            "GBP": 0.79,
            "CAD": 1.36,
            "AUD": 1.52,
            "JPY": 154.50,
            "INR": 83.25
        }
        # Tax jurisdiction rates: (state_rate, county_rate, luxury_threshold, luxury_rate)
        self._tax_tables: Dict[str, Tuple[float, float, float, float]] = {
            "US_CA": (0.0725, 0.0150, 1000.0, 0.010),
            "US_NY": (0.0400, 0.0450, 500.0, 0.015),
            "US_TX": (0.0625, 0.0200, 2000.0, 0.005),
            "US_WA": (0.0650, 0.0300, 1500.0, 0.000),
            "EU_DE": (0.1900, 0.0000, 0.0, 0.000),
            "EU_FR": (0.2000, 0.0000, 0.0, 0.000),
            "IN_KA": (0.0900, 0.0900, 5000.0, 0.020)
        }

    def register_coupon(self, coupon: Coupon) -> None:
        self._coupons[coupon.code.upper()] = coupon

    def set_exchange_rate(self, currency_code: str, rate_to_usd: float) -> None:
        if rate_to_usd <= 0:
            raise ValueError("Exchange rate must be strictly positive")
        self._exchange_rates[currency_code.upper()] = rate_to_usd

    def convert_currency(self, amount: float, from_curr: str, to_curr: str) -> float:
        from_c = from_curr.upper()
        to_c = to_curr.upper()
        if from_c not in self._exchange_rates:
            raise KeyError(f"Unsupported source currency: {from_c}")
        if to_c not in self._exchange_rates:
            raise KeyError(f"Unsupported target currency: {to_c}")

        usd_val = amount / self._exchange_rates[from_c]
        return usd_val * self._exchange_rates[to_c]

    def get_tier_discount_percentage(self, tier: CustomerTier, order_subtotal: float) -> float:
        base_tier_discounts = {
            CustomerTier.BRONZE: 0.0,
            CustomerTier.SILVER: 3.0,
            CustomerTier.GOLD: 7.0,
            CustomerTier.PLATINUM: 12.0
        }
        discount = base_tier_discounts.get(tier, 0.0)

        # High-volume bonus accelerator
        if order_subtotal >= 10000.0:
            discount += 4.0
        elif order_subtotal >= 5000.0:
            discount += 2.0
        elif order_subtotal >= 2000.0:
            discount += 1.0

        return min(discount, 25.0)

    def calculate_line_item_price(
        self,
        item: Item,
        quantity: int,
        customer_tier: CustomerTier
    ) -> Tuple[float, float]:
        """
        Calculates effective unit price and applied volume discount percentage.
        Volume brackets:
          1 - 9   units: 0%
          10 - 49 units: 5%
          50 - 99 units: 10%
          100+    units: 15%
        """
        volume_discount = 0.0
        if quantity >= 100:
            volume_discount = 15.0
        elif quantity >= 50:
            volume_discount = 10.0
        elif quantity >= 10:
            volume_discount = 5.0

        # Tier bonus
        tier_disc = 2.0 if customer_tier in (CustomerTier.GOLD, CustomerTier.PLATINUM) else 0.0
        total_discount_pct = min(volume_discount + tier_disc, 30.0)

        effective_unit_price = item.base_price * (1.0 - total_discount_pct / 100.0)
        return effective_unit_price, total_discount_pct

    def compute_tax(
        self,
        subtotal: float,
        customer: Customer
    ) -> float:
        if customer.tax_exempt:
            return 0.0
        if subtotal <= 0:
            return 0.0

        tax_rule = self._tax_tables.get(customer.jurisdiction_code)
        if not tax_rule:
            # Default fallback tax rate: 5%
            return subtotal * 0.05

        state_rate, county_rate, lux_thresh, lux_rate = tax_rule
        base_tax = subtotal * (state_rate + county_rate)

        # Luxury surcharge if exceeding threshold
        luxury_tax = 0.0
        if lux_thresh > 0 and subtotal > lux_thresh:
            luxury_tax = (subtotal - lux_thresh) * lux_rate

        return base_tax + luxury_tax

    def compute_shipping_fee(
        self,
        total_weight_kg: float,
        destination_zone: str = "DOMESTIC",
        expedited: bool = False
    ) -> float:
        """
        Tiered freight bracket shipping fee calculation.
        """
        if total_weight_kg <= 0:
            return 0.0

        if destination_zone == "DOMESTIC":
            base_fee = 10.0
            per_kg = 1.50
        elif destination_zone == "REGIONAL":
            base_fee = 25.0
            per_kg = 3.20
        else:  # INTERNATIONAL
            base_fee = 65.0
            per_kg = 7.80

        weight_cost = base_fee + (total_weight_kg * per_kg)

        # Heavy pallet fee for freight > 100 kg
        if total_weight_kg > 100.0:
            weight_cost += 50.0

        if expedited:
            weight_cost *= 1.80

        return round(weight_cost, 2)
