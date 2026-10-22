"""
Inventory Service: Manages stock levels across multi-site warehouses.
Provides safety stock formulas, EOQ, reorder point triggers, reservation tracking,
and FIFO batch depletion algorithms.
"""

from __future__ import annotations
import math
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Set

from track_J_software_engineering.target_codebase.models import (
    Item, Warehouse, StockBatch, ItemCategory
)


class OutOfStockException(Exception):
    """Raised when available stock is insufficient to satisfy a fulfillment request."""
    pass


class WarehouseCapacityExceededException(Exception):
    """Raised when incoming pallets exceed facility storage envelope."""
    pass


class InventoryService:
    def __init__(self):
        self._items: Dict[str, Item] = {}
        self._warehouses: Dict[str, Warehouse] = {}
        # Mapping: sku -> warehouse_id -> total on-hand quantity
        self._stock_levels: Dict[str, Dict[str, int]] = {}
        # Mapping: sku -> total reserved quantity across pending orders
        self._reservations: Dict[str, int] = {}
        # Mapping: sku -> List[StockBatch] sorted chronologically for FIFO depletion
        self._batches: Dict[str, List[StockBatch]] = {}

    def register_item(self, item: Item) -> None:
        if item.sku in self._items:
            raise ValueError(f"SKU '{item.sku}' is already registered")
        self._items[item.sku] = item
        self._stock_levels[item.sku] = {}
        self._reservations[item.sku] = 0
        self._batches[item.sku] = []

    def register_warehouse(self, warehouse: Warehouse) -> None:
        if warehouse.warehouse_id in self._warehouses:
            raise ValueError(f"Warehouse ID '{warehouse.warehouse_id}' already registered")
        self._warehouses[warehouse.warehouse_id] = warehouse

    def receive_stock(
        self,
        batch_id: str,
        sku: str,
        warehouse_id: str,
        quantity: int,
        unit_cost: float,
        received_date: Optional[date] = None,
        expiration_date: Optional[date] = None,
        pallets_required: int = 1
    ) -> StockBatch:
        if sku not in self._items:
            raise KeyError(f"Unrecognized SKU: {sku}")
        if warehouse_id not in self._warehouses:
            raise KeyError(f"Unrecognized Warehouse ID: {warehouse_id}")
        if quantity <= 0:
            raise ValueError(f"Received quantity must be positive, got {quantity}")

        wh = self._warehouses[warehouse_id]
        if not wh.active:
            raise RuntimeError(f"Warehouse {warehouse_id} is inactive")
        if not wh.allocate_pallets(pallets_required):
            raise WarehouseCapacityExceededException(
                f"Warehouse {warehouse_id} cannot accommodate {pallets_required} pallets"
            )

        rec_d = received_date or date.today()
        batch = StockBatch(batch_id, sku, warehouse_id, quantity, rec_d, unit_cost, expiration_date)
        self._batches[sku].append(batch)
        # Keep batches ordered by received_date for FIFO
        self._batches[sku].sort(key=lambda b: b.received_date)

        current_wh_stock = self._stock_levels[sku].get(warehouse_id, 0)
        self._stock_levels[sku][warehouse_id] = current_wh_stock + quantity
        return batch

    def get_on_hand(self, sku: str, warehouse_id: Optional[str] = None) -> int:
        if sku not in self._items:
            raise KeyError(f"SKU {sku} not found")
        if warehouse_id is not None:
            return self._stock_levels[sku].get(warehouse_id, 0)
        return sum(self._stock_levels[sku].values())

    def get_reserved(self, sku: str) -> int:
        if sku not in self._items:
            raise KeyError(f"SKU {sku} not found")
        return self._reservations.get(sku, 0)

    def get_available(self, sku: str, warehouse_id: Optional[str] = None) -> int:
        on_hand = self.get_on_hand(sku, warehouse_id)
        if warehouse_id is None:
            reserved = self.get_reserved(sku)
            return max(0, on_hand - reserved)
        return on_hand

    def reserve_stock(self, sku: str, quantity: int) -> bool:
        if quantity <= 0:
            raise ValueError("Reservation quantity must be strictly positive")
        available = self.get_available(sku)
        if available < quantity:
            return False
        self._reservations[sku] = self._reservations.get(sku, 0) + quantity
        return True

    def release_reservation(self, sku: str, quantity: int) -> None:
        if quantity <= 0:
            raise ValueError("Release quantity must be positive")
        current_res = self._reservations.get(sku, 0)
        self._reservations[sku] = max(0, current_res - quantity)

    def fulfill_and_deplete_fifo(self, sku: str, quantity: int, current_date: Optional[date] = None) -> List[Tuple[str, int, float]]:
        """
        Fulfills an allocated quantity using First-In-First-Out (FIFO) batch selection.
        Returns a breakdown of: List of (batch_id, quantity_depleted, batch_unit_cost).
        """
        if quantity <= 0:
            raise ValueError("Fulfill quantity must be positive")
        if self.get_on_hand(sku) < quantity:
            raise OutOfStockException(f"Cannot fulfill {quantity} units of {sku}: insufficient stock")

        today = current_date or date.today()
        depleted_allocations: List[Tuple[str, int, float]] = []
        remaining_to_deplete = quantity

        for batch in list(self._batches[sku]):
            if batch.is_expired(today):
                continue
            if batch.quantity <= 0:
                continue

            take = min(remaining_to_deplete, batch.quantity)
            batch.quantity -= take
            remaining_to_deplete -= take

            # Update warehouse on-hand level
            wh_id = batch.warehouse_id
            self._stock_levels[sku][wh_id] -= take
            depleted_allocations.append((batch.batch_id, take, batch.unit_cost))

            if remaining_to_deplete == 0:
                break

        if remaining_to_deplete > 0:
            raise OutOfStockException(
                f"Failed to satisfy full demand for {sku}. Short by {remaining_to_deplete} units (unexpired stock exhausted)"
            )

        # Release corresponding reservation
        self.release_reservation(sku, quantity)
        # Purge empty batches
        self._batches[sku] = [b for b in self._batches[sku] if b.quantity > 0]
        return depleted_allocations

    def calculate_safety_stock(
        self,
        sku: str,
        service_level_z: float = 1.645,  # 95% service level
        daily_demand_std: float = 5.0
    ) -> int:
        """
        Safety Stock Formula: SS = Z * sigma_L
        where sigma_L = sqrt(L * sigma_d^2) for lead time L.
        """
        item = self._items.get(sku)
        if not item:
            raise KeyError(f"SKU {sku} not found")
        lead_time = max(1, item.lead_time_days)
        sigma_lead_time = math.sqrt(lead_time * (daily_demand_std ** 2))
        return int(math.ceil(service_level_z * sigma_lead_time))

    def calculate_economic_order_quantity(
        self,
        sku: str,
        annual_demand: float,
        order_cost_k: float = 50.0,
        holding_rate_h: float = 0.20
    ) -> int:
        """
        Wilson EOQ Formula: Q* = sqrt((2 * D * K) / H)
        where H = unit_cost * holding_rate_h.
        """
        item = self._items.get(sku)
        if not item:
            raise KeyError(f"SKU {sku} not found")
        h_cost = item.unit_cost * holding_rate_h
        if h_cost <= 0:
            h_cost = 0.01
        eoq = math.sqrt((2.0 * annual_demand * order_cost_k) / h_cost)
        return max(1, int(round(eoq)))

    def evaluate_reorder_trigger(
        self,
        sku: str,
        average_daily_demand: float
    ) -> Tuple[bool, int]:
        """
        Reorder Point (ROP) = (Avg Daily Demand * Lead Time) + Safety Stock
        Returns: (needs_reorder, suggested_order_qty).
        """
        item = self._items.get(sku)
        if not item:
            raise KeyError(f"SKU {sku} not found")
        rop = int(math.ceil(average_daily_demand * item.lead_time_days)) + item.safety_stock
        current_available = self.get_available(sku)
        if current_available <= rop:
            order_qty = max(item.reorder_point, rop - current_available + item.safety_stock)
            return True, order_qty
        return False, 0
