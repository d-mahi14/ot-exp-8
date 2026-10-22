"""
Enterprise Financial & Inventory Management System: Domain Data Models.
Provides strongly-typed dataclasses, enums, validation methods, and domain entities.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum, auto
from typing import List, Dict, Optional, Set, Tuple


class ItemCategory(Enum):
    RAW_MATERIALS = auto()
    COMPONENTS = auto()
    FINISHED_GOODS = auto()
    PACKAGING = auto()
    PERISHABLE = auto()
    CAPITAL_EQUIPMENT = auto()


class CustomerTier(Enum):
    BRONZE = auto()
    SILVER = auto()
    GOLD = auto()
    PLATINUM = auto()


class OrderStatus(Enum):
    DRAFT = auto()
    PENDING_VALIDATION = auto()
    VALIDATED = auto()
    PAYMENT_AUTHORIZED = auto()
    ALLOCATED = auto()
    PACKED = auto()
    SHIPPED = auto()
    DELIVERED = auto()
    CANCELLED = auto()
    RETURNED = auto()


class PaymentMethod(Enum):
    CREDIT_CARD = auto()
    WIRE_TRANSFER = auto()
    NET_30 = auto()
    NET_60 = auto()
    LETTER_OF_CREDIT = auto()


@dataclass
class Item:
    sku: str
    name: str
    category: ItemCategory
    unit_cost: float
    base_price: float
    reorder_point: int
    safety_stock: int
    lead_time_days: int
    weight_kg: float
    active: bool = True
    attributes: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.unit_cost < 0:
            raise ValueError(f"Unit cost cannot be negative: {self.unit_cost}")
        if self.base_price < self.unit_cost:
            raise ValueError(f"Base price ({self.base_price}) cannot be below unit cost ({self.unit_cost})")
        if self.weight_kg <= 0:
            raise ValueError(f"Weight must be strictly positive: {self.weight_kg}")
        if self.lead_time_days < 0:
            raise ValueError("Lead time cannot be negative")

    @property
    def gross_margin(self) -> float:
        if self.base_price == 0:
            return 0.0
        return (self.base_price - self.unit_cost) / self.base_price

    @property
    def markup_percentage(self) -> float:
        if self.unit_cost == 0:
            return 0.0
        return ((self.base_price - self.unit_cost) / self.unit_cost) * 100.0


@dataclass
class Customer:
    customer_id: str
    company_name: str
    tier: CustomerTier
    credit_limit: float
    current_outstanding: float = 0.0
    tax_exempt: bool = False
    jurisdiction_code: str = "US_CA"
    risk_score: float = 0.10  # 0.0 = zero risk, 1.0 = extreme risk
    created_at: datetime = field(default_factory=datetime.now)

    def has_available_credit(self, amount: float) -> bool:
        if amount < 0:
            raise ValueError("Evaluation amount cannot be negative")
        return (self.current_outstanding + amount) <= self.credit_limit

    def adjust_outstanding(self, amount: float) -> None:
        new_val = self.current_outstanding + amount
        if new_val < -1e-6:
            raise ValueError("Outstanding balance cannot become negative")
        self.current_outstanding = max(0.0, new_val)


@dataclass
class OrderLine:
    line_number: int
    sku: str
    quantity: int
    unit_price: float
    discount_pct: float = 0.0
    allocated: bool = False

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"Line quantity must be positive, got {self.quantity}")
        if self.unit_price < 0:
            raise ValueError("Unit price cannot be negative")
        if not (0.0 <= self.discount_pct <= 100.0):
            raise ValueError("Discount percentage must be between 0.0 and 100.0")

    @property
    def net_unit_price(self) -> float:
        return self.unit_price * (1.0 - self.discount_pct / 100.0)

    @property
    def subtotal(self) -> float:
        return self.quantity * self.net_unit_price


@dataclass
class Order:
    order_id: str
    customer_id: str
    lines: List[OrderLine] = field(default_factory=list)
    status: OrderStatus = OrderStatus.DRAFT
    payment_method: PaymentMethod = PaymentMethod.CREDIT_CARD
    order_date: datetime = field(default_factory=datetime.now)
    shipping_address: str = "Default Distribution Hub"
    currency: str = "USD"
    notes: Optional[str] = None
    applied_coupon: Optional[str] = None

    def add_line(self, sku: str, quantity: int, unit_price: float, discount_pct: float = 0.0) -> OrderLine:
        if self.status != OrderStatus.DRAFT:
            raise RuntimeError(f"Cannot add lines to order in status {self.status.name}")
        line_num = len(self.lines) + 1
        line = OrderLine(line_num, sku, quantity, unit_price, discount_pct)
        self.lines.append(line)
        return line

    @property
    def subtotal(self) -> float:
        return sum(line.subtotal for line in self.lines)

    @property
    def total_weight_kg(self) -> float:
        # Default unit weight weight factor per item line if not cross-referenced
        return sum(line.quantity * 0.5 for line in self.lines)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def total_item_units(self) -> int:
        return sum(line.quantity for line in self.lines)


@dataclass
class Invoice:
    invoice_id: str
    order_id: str
    customer_id: str
    subtotal: float
    tax_amount: float
    shipping_fee: float
    total_amount: float
    issued_date: date = field(default_factory=date.today)
    due_date: Optional[date] = None
    paid: bool = False

    def verify_reconciliation(self, tolerance: float = 1e-4) -> bool:
        expected = self.subtotal + self.tax_amount + self.shipping_fee
        return abs(self.total_amount - expected) <= tolerance


@dataclass
class Warehouse:
    warehouse_id: str
    location_name: str
    capacity_pallets: int
    occupied_pallets: int = 0
    temperature_controlled: bool = False
    active: bool = True

    def available_capacity(self) -> int:
        return max(0, self.capacity_pallets - self.occupied_pallets)

    def allocate_pallets(self, count: int) -> bool:
        if count <= 0:
            raise ValueError("Pallet count must be positive")
        if self.occupied_pallets + count > self.capacity_pallets:
            return False
        self.occupied_pallets += count
        return True

    def release_pallets(self, count: int) -> None:
        if count <= 0:
            raise ValueError("Pallet count must be positive")
        self.occupied_pallets = max(0, self.occupied_pallets - count)


@dataclass
class StockBatch:
    batch_id: str
    sku: str
    warehouse_id: str
    quantity: int
    received_date: date
    unit_cost: float
    expiration_date: Optional[date] = None

    def is_expired(self, current_date: Optional[date] = None) -> bool:
        if self.expiration_date is None:
            return False
        now = current_date or date.today()
        return self.expiration_date < now
