"""
Order Processing Engine: End-to-end order workflow lifecycle state machine.
Enforces credit limit verification, fraud scoring heuristics, inventory reservation,
and automated invoice generation.
"""

from __future__ import annotations
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

from track_J_software_engineering.target_codebase.models import (
    Customer, Order, OrderLine, OrderStatus, Invoice, PaymentMethod
)
from track_J_software_engineering.target_codebase.inventory_service import (
    InventoryService, OutOfStockException
)
from track_J_software_engineering.target_codebase.pricing_engine import PricingEngine


class OrderValidationException(Exception):
    """Raised when an order fails business rule validation."""
    pass


class CreditLimitExceededException(Exception):
    """Raised when a customer's available credit line is insufficient."""
    pass


class FraudAlertException(Exception):
    """Raised when an order triggers anti-fraud risk thresholds."""
    pass


class OrderProcessor:
    def __init__(self, inventory_service: InventoryService, pricing_engine: PricingEngine):
        self.inventory_service = inventory_service
        self.pricing_engine = pricing_engine
        self._customers: Dict[str, Customer] = {}
        self._orders: Dict[str, Order] = {}
        self._invoices: Dict[str, Invoice] = {}
        self._audit_trail: List[Dict[str, str]] = []

    def register_customer(self, customer: Customer) -> None:
        self._customers[customer.customer_id] = customer

    def record_audit(self, order_id: str, event: str, details: str) -> None:
        self._audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "order_id": order_id,
            "event": event,
            "details": details
        })

    def get_order(self, order_id: str) -> Order:
        if order_id not in self._orders:
            raise KeyError(f"Order {order_id} not found")
        return self._orders[order_id]

    def get_invoice(self, invoice_id: str) -> Invoice:
        if invoice_id not in self._invoices:
            raise KeyError(f"Invoice {invoice_id} not found")
        return self._invoices[invoice_id]

    def create_order(
        self,
        order_id: str,
        customer_id: str,
        payment_method: PaymentMethod = PaymentMethod.CREDIT_CARD,
        currency: str = "USD"
    ) -> Order:
        if customer_id not in self._customers:
            raise KeyError(f"Customer {customer_id} does not exist")
        if order_id in self._orders:
            raise ValueError(f"Order {order_id} already exists")

        order = Order(order_id=order_id, customer_id=customer_id, payment_method=payment_method, currency=currency)
        self._orders[order_id] = order
        self.record_audit(order_id, "ORDER_CREATED", f"Initialized draft for customer {customer_id}")
        return order

    def evaluate_fraud_risk(self, order: Order, customer: Customer) -> float:
        """
        Calculates a composite fraud risk score in [0.0, 1.0].
        Triggers alert if risk > 0.70.
        """
        score = customer.risk_score

        # Rule 1: High velocity / large amount check
        if order.subtotal > 20000.0:
            score += 0.35
        elif order.subtotal > 5000.0:
            score += 0.15

        # Rule 2: Payment method risk weighting
        if order.payment_method == PaymentMethod.WIRE_TRANSFER:
            score += 0.05
        elif order.payment_method == PaymentMethod.CREDIT_CARD and order.subtotal > 8000.0:
            score += 0.20

        # Rule 3: Line count anomalies
        if order.line_count > 50:
            score += 0.10

        return min(1.0, score)

    def validate_and_reserve(self, order_id: str) -> bool:
        order = self.get_order(order_id)
        if order.status != OrderStatus.DRAFT:
            raise OrderValidationException(f"Order {order_id} is in status {order.status.name}, must be DRAFT")
        if order.line_count == 0:
            raise OrderValidationException(f"Order {order_id} has no line items")

        customer = self._customers[order.customer_id]

        # Fraud screening
        risk_score = self.evaluate_fraud_risk(order, customer)
        if risk_score > 0.70:
            order.status = OrderStatus.CANCELLED
            self.record_audit(order_id, "FRAUD_REJECTION", f"Risk score {risk_score:.2f} exceeded safety threshold")
            raise FraudAlertException(f"Order {order_id} rejected due to high fraud risk score: {risk_score:.2f}")

        # Credit validation
        subtotal = order.subtotal
        if order.payment_method in (PaymentMethod.NET_30, PaymentMethod.NET_60):
            if not customer.has_available_credit(subtotal):
                raise CreditLimitExceededException(
                    f"Credit limit exceeded: customer limit={customer.credit_limit}, outstanding={customer.current_outstanding}, order={subtotal}"
                )

        # Inventory reservation check
        for line in order.lines:
            avail = self.inventory_service.get_available(line.sku)
            if avail < line.quantity:
                raise OutOfStockException(
                    f"Insufficient stock for SKU {line.sku}: available {avail}, requested {line.quantity}"
                )

        # Execute reservations
        for line in order.lines:
            reserved = self.inventory_service.reserve_stock(line.sku, line.quantity)
            if not reserved:
                # Rollback previously reserved lines
                for r_line in order.lines:
                    if r_line.allocated:
                        self.inventory_service.release_reservation(r_line.sku, r_line.quantity)
                        r_line.allocated = False
                raise OutOfStockException(f"Concurrency conflict while reserving {line.sku}")
            line.allocated = True

        order.status = OrderStatus.VALIDATED
        self.record_audit(order_id, "ORDER_VALIDATED", f"Reserved stock for {order.line_count} lines")
        return True

    def authorize_payment(self, order_id: str) -> Invoice:
        order = self.get_order(order_id)
        if order.status != OrderStatus.VALIDATED:
            raise OrderValidationException(f"Order {order_id} must be VALIDATED before payment authorization")

        customer = self._customers[order.customer_id]
        subtotal = order.subtotal

        # Apply coupon if specified
        discount_amount = 0.0
        if order.applied_coupon:
            coupon = self.pricing_engine._coupons.get(order.applied_coupon.upper())
            if coupon and coupon.is_valid(subtotal):
                discount_amount = coupon.compute_discount(subtotal)

        discounted_subtotal = max(0.0, subtotal - discount_amount)
        tax = self.pricing_engine.compute_tax(discounted_subtotal, customer)
        shipping = self.pricing_engine.compute_shipping_fee(order.total_weight_kg)
        total_amount = round(discounted_subtotal + tax + shipping, 2)

        # Adjust customer outstanding balance if credit-based
        if order.payment_method in (PaymentMethod.NET_30, PaymentMethod.NET_60):
            customer.adjust_outstanding(total_amount)

        invoice_id = f"INV-{uuid.uuid4().hex[:8].upper()}"
        due_days = 30 if order.payment_method == PaymentMethod.NET_30 else (60 if order.payment_method == PaymentMethod.NET_60 else 0)
        due_d = date.today() + timedelta(days=due_days)

        invoice = Invoice(
            invoice_id=invoice_id,
            order_id=order_id,
            customer_id=customer.customer_id,
            subtotal=discounted_subtotal,
            tax_amount=tax,
            shipping_fee=shipping,
            total_amount=total_amount,
            issued_date=date.today(),
            due_date=due_d,
            paid=(due_days == 0)
        )
        self._invoices[invoice_id] = invoice

        order.status = OrderStatus.PAYMENT_AUTHORIZED
        self.record_audit(order_id, "PAYMENT_AUTHORIZED", f"Generated invoice {invoice_id} for ${total_amount:.2f}")
        return invoice

    def fulfill_and_ship_order(self, order_id: str) -> Dict[str, List[Tuple[str, int, float]]]:
        order = self.get_order(order_id)
        if order.status != OrderStatus.PAYMENT_AUTHORIZED:
            raise OrderValidationException(f"Order {order_id} must be PAYMENT_AUTHORIZED before shipment")

        depletion_manifest: Dict[str, List[Tuple[str, int, float]]] = {}
        for line in order.lines:
            allocations = self.inventory_service.fulfill_and_deplete_fifo(line.sku, line.quantity)
            depletion_manifest[line.sku] = allocations

        order.status = OrderStatus.SHIPPED
        self.record_audit(order_id, "ORDER_SHIPPED", f"Dispatched {order.line_count} items")
        return depletion_manifest

    def cancel_order(self, order_id: str, reason: str = "Customer requested cancellation") -> None:
        order = self.get_order(order_id)
        if order.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED):
            raise OrderValidationException(f"Cannot cancel shipped or delivered order {order_id}")

        # If reservations were placed, release them
        if order.status in (OrderStatus.VALIDATED, OrderStatus.PAYMENT_AUTHORIZED):
            for line in order.lines:
                if line.allocated:
                    self.inventory_service.release_reservation(line.sku, line.quantity)
                    line.allocated = False

        order.status = OrderStatus.CANCELLED
        self.record_audit(order_id, "ORDER_CANCELLED", reason)
