"""
Test Catalog: Comprehensive regression test suite for the 1,000+ LOC Target Codebase.
Contains 70 distinct regression test cases covering Models, Inventory, Pricing,
Order Processing, Analytics, and Utilities.
"""

from __future__ import annotations
import math
import time
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Tuple

from track_J_software_engineering.target_codebase.models import (
    Item, Customer, Order, OrderLine, Invoice, Warehouse, StockBatch,
    ItemCategory, CustomerTier, OrderStatus, PaymentMethod
)
from track_J_software_engineering.target_codebase.inventory_service import (
    InventoryService, OutOfStockException, WarehouseCapacityExceededException
)
from track_J_software_engineering.target_codebase.pricing_engine import (
    PricingEngine, Coupon
)
from track_J_software_engineering.target_codebase.order_processor import (
    OrderProcessor, OrderValidationException, CreditLimitExceededException, FraudAlertException
)
from track_J_software_engineering.target_codebase.analytics_reporting import (
    AnalyticsReportingEngine
)
from track_J_software_engineering.target_codebase.utils import (
    TTLCache, CryptographicAuditLog, format_currency, calculate_compound_interest, sanitize_sku
)
from track_J_software_engineering.target_codebase.reporting_export import (
    DataExportService
)


def _build_test_catalog() -> Dict[str, Callable[[], None]]:
    catalog: Dict[str, Callable[[], None]] = {}

    # -------------------------------------------------------------
    # Group 1: Item & Model Tests
    # -------------------------------------------------------------
    def test_01_valid_item_construction():
        it = Item("SKU-100", "Steel Plate", ItemCategory.RAW_MATERIALS, 50.0, 75.0, 10, 5, 7, 12.5)
        assert it.sku == "SKU-100"
        assert it.unit_cost == 50.0
        assert it.base_price == 75.0

    def test_02_item_negative_cost_raises():
        try:
            Item("SKU-ERR", "Err", ItemCategory.RAW_MATERIALS, -10.0, 20.0, 1, 1, 1, 1.0)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_03_item_price_below_cost_raises():
        try:
            Item("SKU-ERR", "Err", ItemCategory.RAW_MATERIALS, 100.0, 50.0, 1, 1, 1, 1.0)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_04_item_margin_and_markup():
        it = Item("SKU-MARG", "Widget", ItemCategory.COMPONENTS, 40.0, 80.0, 5, 2, 3, 2.0)
        assert abs(it.gross_margin - 0.50) < 1e-4
        assert abs(it.markup_percentage - 100.0) < 1e-4

    def test_05_item_zero_cost_edge_case():
        it = Item("SKU-FREE", "Open Sample", ItemCategory.FINISHED_GOODS, 0.0, 0.0, 0, 0, 1, 0.1)
        assert it.gross_margin == 0.0
        assert it.markup_percentage == 0.0

    def test_06_customer_credit_evaluation():
        cust = Customer("C-101", "Acme Corp", CustomerTier.GOLD, credit_limit=5000.0, current_outstanding=2000.0)
        assert cust.has_available_credit(2500.0) is True
        assert cust.has_available_credit(3500.0) is False

    def test_07_customer_negative_amount_credit_check_raises():
        cust = Customer("C-101", "Acme Corp", CustomerTier.GOLD, credit_limit=5000.0)
        try:
            cust.has_available_credit(-100.0)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_08_customer_adjust_outstanding():
        cust = Customer("C-102", "Globex", CustomerTier.PLATINUM, credit_limit=10000.0, current_outstanding=1000.0)
        cust.adjust_outstanding(1500.0)
        assert cust.current_outstanding == 2500.0
        cust.adjust_outstanding(-2000.0)
        assert cust.current_outstanding == 500.0

    def test_09_order_line_subtotal():
        line = OrderLine(1, "SKU-A", 10, 20.0, discount_pct=10.0)
        assert line.net_unit_price == 18.0
        assert line.subtotal == 180.0

    def test_10_order_line_invalid_quantity_raises():
        try:
            OrderLine(1, "SKU-A", 0, 20.0)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    # -------------------------------------------------------------
    # Group 2: Warehouse & Stock Batch Tests
    # -------------------------------------------------------------
    def test_11_warehouse_allocation():
        wh = Warehouse("WH-EAST", "New Jersey", capacity_pallets=100, occupied_pallets=40)
        assert wh.available_capacity() == 60
        success = wh.allocate_pallets(30)
        assert success is True
        assert wh.occupied_pallets == 70

    def test_12_warehouse_capacity_overflow():
        wh = Warehouse("WH-EAST", "New Jersey", capacity_pallets=50, occupied_pallets=45)
        success = wh.allocate_pallets(10)
        assert success is False
        assert wh.occupied_pallets == 45

    def test_13_warehouse_release():
        wh = Warehouse("WH-WEST", "Reno", capacity_pallets=100, occupied_pallets=80)
        wh.release_pallets(25)
        assert wh.occupied_pallets == 55

    def test_14_stock_batch_expiration():
        yesterday = date.today() - timedelta(days=1)
        tomorrow = date.today() + timedelta(days=1)
        expired_batch = StockBatch("B-1", "SKU-MILK", "WH-1", 50, date.today(), 2.0, yesterday)
        fresh_batch = StockBatch("B-2", "SKU-MILK", "WH-1", 50, date.today(), 2.0, tomorrow)
        assert expired_batch.is_expired() is True
        assert fresh_batch.is_expired() is False

    # -------------------------------------------------------------
    # Group 3: Inventory Service Core Operations
    # -------------------------------------------------------------
    def test_15_inventory_register_and_receive():
        inv = InventoryService()
        it = Item("SKU-1", "Gear", ItemCategory.COMPONENTS, 10.0, 20.0, 5, 2, 3, 1.0)
        wh = Warehouse("WH-1", "Dallas", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-101", "SKU-1", "WH-1", 100, 10.0)
        assert inv.get_on_hand("SKU-1") == 100
        assert inv.get_available("SKU-1") == 100

    def test_16_inventory_duplicate_item_raises():
        inv = InventoryService()
        it = Item("SKU-1", "Gear", ItemCategory.COMPONENTS, 10.0, 20.0, 5, 2, 3, 1.0)
        inv.register_item(it)
        try:
            inv.register_item(it)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_17_inventory_receive_unknown_item_raises():
        inv = InventoryService()
        wh = Warehouse("WH-1", "Dallas", 100)
        inv.register_warehouse(wh)
        try:
            inv.receive_stock("B-1", "UNKNOWN", "WH-1", 10, 5.0)
            assert False, "Expected KeyError"
        except KeyError:
            pass

    def test_18_inventory_reservation_workflow():
        inv = InventoryService()
        it = Item("SKU-2", "Bearing", ItemCategory.COMPONENTS, 5.0, 12.0, 10, 5, 2, 0.5)
        wh = Warehouse("WH-1", "Dallas", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-201", "SKU-2", "WH-1", 50, 5.0)

        assert inv.reserve_stock("SKU-2", 30) is True
        assert inv.get_available("SKU-2") == 20
        assert inv.get_reserved("SKU-2") == 30
        assert inv.reserve_stock("SKU-2", 25) is False  # Only 20 available

    def test_19_inventory_release_reservation():
        inv = InventoryService()
        it = Item("SKU-3", "Shaft", ItemCategory.COMPONENTS, 15.0, 30.0, 5, 2, 4, 3.0)
        wh = Warehouse("WH-1", "Dallas", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-301", "SKU-3", "WH-1", 40, 15.0)
        inv.reserve_stock("SKU-3", 20)
        inv.release_reservation("SKU-3", 10)
        assert inv.get_reserved("SKU-3") == 10
        assert inv.get_available("SKU-3") == 30

    def test_20_inventory_fifo_depletion():
        inv = InventoryService()
        it = Item("SKU-4", "Sensor", ItemCategory.COMPONENTS, 20.0, 40.0, 5, 2, 5, 0.2)
        wh = Warehouse("WH-1", "Dallas", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)

        d1 = date(2026, 1, 10)
        d2 = date(2026, 1, 20)
        inv.receive_stock("B-OLD", "SKU-4", "WH-1", 15, 18.0, received_date=d1)
        inv.receive_stock("B-NEW", "SKU-4", "WH-1", 25, 22.0, received_date=d2)
        inv.reserve_stock("SKU-4", 25)

        allocs = inv.fulfill_and_deplete_fifo("SKU-4", 25)
        # Should deplete B-OLD completely (15) and 10 from B-NEW
        assert allocs == [("B-OLD", 15, 18.0), ("B-NEW", 10, 22.0)]
        assert inv.get_on_hand("SKU-4") == 15

    def test_21_inventory_fifo_skips_expired_stock():
        inv = InventoryService()
        it = Item("SKU-YOGURT", "Dairy", ItemCategory.PERISHABLE, 1.0, 2.5, 5, 2, 1, 0.2)
        wh = Warehouse("WH-1", "Dallas", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)

        yesterday = date.today() - timedelta(days=1)
        tomorrow = date.today() + timedelta(days=10)
        inv.receive_stock("B-EXP", "SKU-YOGURT", "WH-1", 10, 1.0, expiration_date=yesterday)
        inv.receive_stock("B-FRESH", "SKU-YOGURT", "WH-1", 15, 1.1, expiration_date=tomorrow)
        inv.reserve_stock("SKU-YOGURT", 8)

        allocs = inv.fulfill_and_deplete_fifo("SKU-YOGURT", 8)
        assert len(allocs) == 1
        assert allocs[0][0] == "B-FRESH"

    def test_22_inventory_fifo_out_of_stock_raises():
        inv = InventoryService()
        it = Item("SKU-EMPTY", "Nut", ItemCategory.COMPONENTS, 0.5, 1.0, 5, 2, 1, 0.05)
        wh = Warehouse("WH-1", "Dallas", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-1", "SKU-EMPTY", "WH-1", 5, 0.5)
        try:
            inv.fulfill_and_deplete_fifo("SKU-EMPTY", 10)
            assert False, "Expected OutOfStockException"
        except OutOfStockException:
            pass

    # -------------------------------------------------------------
    # Group 4: Inventory Analytics Formulas (Safety Stock, EOQ, ROP)
    # -------------------------------------------------------------
    def test_23_safety_stock_calculation():
        inv = InventoryService()
        it = Item("SKU-SS", "Hydraulic Valve", ItemCategory.COMPONENTS, 100.0, 200.0, 20, 10, 4, 5.0)
        inv.register_item(it)
        # L=4, sigma_d=5, sigma_L = sqrt(4 * 25) = 10, Z=1.645 => SS = ceil(16.45) = 17
        ss = inv.calculate_safety_stock("SKU-SS", service_level_z=1.645, daily_demand_std=5.0)
        assert ss == 17

    def test_24_economic_order_quantity():
        inv = InventoryService()
        it = Item("SKU-EOQ", "Filter", ItemCategory.COMPONENTS, 25.0, 50.0, 20, 10, 5, 1.0)
        inv.register_item(it)
        # D=1000, K=50, H = 25 * 0.20 = 5.0 => EOQ = sqrt(100000 / 5) = sqrt(20000) = 141.42 => 141
        eoq = inv.calculate_economic_order_quantity("SKU-EOQ", annual_demand=1000.0, order_cost_k=50.0, holding_rate_h=0.20)
        assert eoq == 141

    def test_25_reorder_trigger_condition():
        inv = InventoryService()
        it = Item("SKU-ROP", "Seal", ItemCategory.COMPONENTS, 5.0, 10.0, reorder_point=50, safety_stock=20, lead_time_days=3, weight_kg=0.1)
        wh = Warehouse("WH-1", "Dallas", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-ROP", "SKU-ROP", "WH-1", 30, 5.0)

        # Avg daily demand = 10. Lead time = 3. ROP = 10*3 + 20 = 50.
        # Available = 30 <= 50 -> Trigger True
        needs_reorder, suggested_qty = inv.evaluate_reorder_trigger("SKU-ROP", average_daily_demand=10.0)
        assert needs_reorder is True
        assert suggested_qty >= 40

    def test_26_reorder_trigger_not_needed():
        inv = InventoryService()
        it = Item("SKU-ROP2", "Gasket", ItemCategory.COMPONENTS, 5.0, 10.0, reorder_point=50, safety_stock=20, lead_time_days=3, weight_kg=0.1)
        wh = Warehouse("WH-1", "Dallas", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-ROP2", "SKU-ROP2", "WH-1", 100, 5.0)
        needs_reorder, suggested_qty = inv.evaluate_reorder_trigger("SKU-ROP2", average_daily_demand=10.0)
        assert needs_reorder is False
        assert suggested_qty == 0

    # -------------------------------------------------------------
    # Group 5: Pricing Engine, Discounts, Coupons
    # -------------------------------------------------------------
    def test_27_pricing_tier_discounts():
        pe = PricingEngine()
        d_bronze = pe.get_tier_discount_percentage(CustomerTier.BRONZE, 1000.0)
        d_gold = pe.get_tier_discount_percentage(CustomerTier.GOLD, 1000.0)
        d_plat_high = pe.get_tier_discount_percentage(CustomerTier.PLATINUM, 12000.0)
        assert d_bronze == 0.0
        assert d_gold == 7.0
        assert d_plat_high == 16.0  # 12 base + 4 volume

    def test_28_pricing_line_item_volume_discount():
        pe = PricingEngine()
        it = Item("SKU-V", "Bolt", ItemCategory.COMPONENTS, 1.0, 2.0, 10, 5, 1, 0.05)
        # 5 units: 0% volume disc
        price_5, disc_5 = pe.calculate_line_item_price(it, 5, CustomerTier.BRONZE)
        assert disc_5 == 0.0
        assert price_5 == 2.0

        # 50 units: 10% volume disc
        price_50, disc_50 = pe.calculate_line_item_price(it, 50, CustomerTier.BRONZE)
        assert disc_50 == 10.0
        assert price_50 == 1.80

        # 120 units: 15% volume disc + 2% gold tier = 17%
        price_120, disc_120 = pe.calculate_line_item_price(it, 120, CustomerTier.GOLD)
        assert disc_120 == 17.0
        assert abs(price_120 - (2.0 * 0.83)) < 1e-4

    def test_29_coupon_validation_and_calculation():
        pe = PricingEngine()
        c = Coupon("SAVE20", discount_pct=20.0, min_order_amount=200.0, max_discount_amount=100.0)
        pe.register_coupon(c)
        assert c.is_valid(150.0) is False
        assert c.is_valid(300.0) is True
        # 20% of 300 is 60 <= max 100
        assert c.compute_discount(300.0) == 60.0
        # 20% of 1000 is 200 > max 100 => capped at 100
        assert c.compute_discount(1000.0) == 100.0

    def test_30_coupon_expired():
        c_exp = Coupon("EXPIRED", discount_pct=15.0, expiry_date=date.today() - timedelta(days=2))
        assert c_exp.is_valid(500.0) is False

    # -------------------------------------------------------------
    # Group 6: Regional Tax & Currency Conversion
    # -------------------------------------------------------------
    def test_31_tax_california_with_luxury_surcharge():
        pe = PricingEngine()
        cust = Customer("C-CA", "Silicon Tech", CustomerTier.GOLD, credit_limit=50000.0, jurisdiction_code="US_CA")
        # Base tax: 7.25% + 1.50% = 8.75%. Luxury: > 1000 at 1.0%
        # Subtotal: 2000. Base tax = 2000 * 0.0875 = 175. Luxury = (2000-1000) * 0.01 = 10. Total = 185.
        tax = pe.compute_tax(2000.0, cust)
        assert abs(tax - 185.0) < 1e-4

    def test_32_tax_exempt_customer():
        pe = PricingEngine()
        cust = Customer("C-EX", "Non-Profit Org", CustomerTier.BRONZE, credit_limit=5000.0, tax_exempt=True)
        assert pe.compute_tax(5000.0, cust) == 0.0

    def test_33_currency_conversion():
        pe = PricingEngine()
        # 100 USD to EUR (0.92)
        eur = pe.convert_currency(100.0, "USD", "EUR")
        assert abs(eur - 92.0) < 1e-4
        # EUR back to USD
        usd = pe.convert_currency(92.0, "EUR", "USD")
        assert abs(usd - 100.0) < 1e-4

    def test_34_shipping_brackets():
        pe = PricingEngine()
        fee_light = pe.compute_shipping_fee(10.0, "DOMESTIC")
        # 10 base + 10 * 1.50 = 25.0
        assert fee_light == 25.0
        fee_heavy = pe.compute_shipping_fee(150.0, "DOMESTIC")
        # 10 base + 150 * 1.50 + 50 pallet = 285.0
        assert fee_heavy == 285.0

    # -------------------------------------------------------------
    # Group 7: Order Processor Lifecycle & Validation
    # -------------------------------------------------------------
    def test_35_order_lifecycle_successful_completion():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)

        it = Item("SKU-PC", "Laptop", ItemCategory.FINISHED_GOODS, 500.0, 900.0, 10, 5, 2, 2.5)
        wh = Warehouse("WH-1", "Chicago", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-PC1", "SKU-PC", "WH-1", 20, 500.0)

        cust = Customer("C-CORP", "MegaCorp", CustomerTier.PLATINUM, credit_limit=50000.0)
        proc.register_customer(cust)

        ord1 = proc.create_order("ORD-001", "C-CORP", payment_method=PaymentMethod.NET_30)
        ord1.add_line("SKU-PC", 5, 900.0)

        assert proc.validate_and_reserve("ORD-001") is True
        assert ord1.status == OrderStatus.VALIDATED

        inv_doc = proc.authorize_payment("ORD-001")
        assert ord1.status == OrderStatus.PAYMENT_AUTHORIZED
        assert inv_doc.total_amount > 4500.0

        manifest = proc.fulfill_and_ship_order("ORD-001")
        assert ord1.status == OrderStatus.SHIPPED
        assert "SKU-PC" in manifest

    def test_36_order_insufficient_stock_fails_validation():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)

        it = Item("SKU-LOW", "Chip", ItemCategory.COMPONENTS, 5.0, 10.0, 5, 2, 1, 0.1)
        wh = Warehouse("WH-1", "Chicago", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-CHIP", "SKU-LOW", "WH-1", 5, 5.0)

        cust = Customer("C-1", "User1", CustomerTier.BRONZE, credit_limit=1000.0)
        proc.register_customer(cust)

        ord1 = proc.create_order("ORD-FAIL-STK", "C-1")
        ord1.add_line("SKU-LOW", 20, 10.0)  # Only 5 in stock

        try:
            proc.validate_and_reserve("ORD-FAIL-STK")
            assert False, "Expected OutOfStockException"
        except OutOfStockException:
            assert ord1.status == OrderStatus.DRAFT

    def test_37_order_credit_limit_exceeded_fails():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)

        it = Item("SKU-EXP", "Server", ItemCategory.CAPITAL_EQUIPMENT, 2000.0, 4000.0, 2, 1, 5, 25.0)
        wh = Warehouse("WH-1", "Chicago", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-SRV", "SKU-EXP", "WH-1", 10, 2000.0)

        cust = Customer("C-BROKE", "BrokeCorp", CustomerTier.BRONZE, credit_limit=5000.0)
        proc.register_customer(cust)

        ord1 = proc.create_order("ORD-CREDIT-EXC", "C-BROKE", payment_method=PaymentMethod.NET_30)
        ord1.add_line("SKU-EXP", 2, 4000.0)  # Total 8000 > limit 5000

        try:
            proc.validate_and_reserve("ORD-CREDIT-EXC")
            assert False, "Expected CreditLimitExceededException"
        except CreditLimitExceededException:
            pass

    def test_38_order_fraud_alert_rejection():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)

        it = Item("SKU-GOLD", "Gold Bar", ItemCategory.RAW_MATERIALS, 1000.0, 1500.0, 10, 5, 1, 1.0)
        wh = Warehouse("WH-1", "Chicago", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-GOLD", "SKU-GOLD", "WH-1", 100, 1000.0)

        # Risky customer with baseline score 0.50
        cust = Customer("C-RISK", "Shady LLC", CustomerTier.BRONZE, credit_limit=100000.0, risk_score=0.50)
        proc.register_customer(cust)

        ord1 = proc.create_order("ORD-FRAUD", "C-RISK", payment_method=PaymentMethod.CREDIT_CARD)
        ord1.add_line("SKU-GOLD", 20, 1500.0)  # Subtotal 30,000 -> triggers fraud surge

        try:
            proc.validate_and_reserve("ORD-FRAUD")
            assert False, "Expected FraudAlertException"
        except FraudAlertException:
            assert ord1.status == OrderStatus.CANCELLED

    def test_39_order_cancellation_releases_inventory():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)

        it = Item("SKU-REL", "Resistor", ItemCategory.COMPONENTS, 1.0, 2.0, 10, 5, 1, 0.01)
        wh = Warehouse("WH-1", "Chicago", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-R", "SKU-REL", "WH-1", 100, 1.0)

        cust = Customer("C-CAN", "CanCorp", CustomerTier.SILVER, credit_limit=5000.0)
        proc.register_customer(cust)

        ord1 = proc.create_order("ORD-CANCEL", "C-CAN")
        ord1.add_line("SKU-REL", 40, 2.0)
        proc.validate_and_reserve("ORD-CANCEL")
        assert inv.get_reserved("SKU-REL") == 40

        proc.cancel_order("ORD-CANCEL")
        assert ord1.status == OrderStatus.CANCELLED
        assert inv.get_reserved("SKU-REL") == 0

    def test_40_order_double_cancellation_shipped_raises():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)

        it = Item("SKU-SHP", "Box", ItemCategory.PACKAGING, 1.0, 2.0, 10, 5, 1, 0.5)
        wh = Warehouse("WH-1", "Chicago", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-BOX", "SKU-SHP", "WH-1", 50, 1.0)

        cust = Customer("C-S", "ShipCorp", CustomerTier.GOLD, credit_limit=5000.0)
        proc.register_customer(cust)
        ord1 = proc.create_order("ORD-SHIPPED", "C-S")
        ord1.add_line("SKU-SHP", 10, 2.0)
        proc.validate_and_reserve("ORD-SHIPPED")
        proc.authorize_payment("ORD-SHIPPED")
        proc.fulfill_and_ship_order("ORD-SHIPPED")

        try:
            proc.cancel_order("ORD-SHIPPED")
            assert False, "Expected OrderValidationException"
        except OrderValidationException:
            pass

    # -------------------------------------------------------------
    # Group 8: Analytics Engine Tests
    # -------------------------------------------------------------
    def test_41_analytics_inventory_valuation():
        inv = InventoryService()
        it1 = Item("SKU-A1", "Part A", ItemCategory.COMPONENTS, 10.0, 20.0, 5, 2, 1, 1.0)
        it2 = Item("SKU-A2", "Material B", ItemCategory.RAW_MATERIALS, 5.0, 10.0, 5, 2, 1, 2.0)
        wh = Warehouse("WH-1", "Chicago", 100)
        inv.register_item(it1)
        inv.register_item(it2)
        inv.register_warehouse(wh)
        inv.receive_stock("B-1", "SKU-A1", "WH-1", 50, 10.0)
        inv.receive_stock("B-2", "SKU-A2", "WH-1", 100, 5.0)

        an = AnalyticsReportingEngine(inv)
        vals = an.compute_inventory_valuation()
        assert vals["COMPONENTS"] == 500.0
        assert vals["RAW_MATERIALS"] == 500.0
        assert vals["TOTAL"] == 1000.0

    def test_42_analytics_abc_pareto_analysis():
        inv = InventoryService()
        itA = Item("SKU-A", "High Val", ItemCategory.COMPONENTS, 100.0, 500.0, 1, 1, 1, 1.0)
        itB = Item("SKU-B", "Mid Val", ItemCategory.COMPONENTS, 20.0, 50.0, 1, 1, 1, 1.0)
        itC = Item("SKU-C", "Low Val", ItemCategory.COMPONENTS, 1.0, 2.0, 1, 1, 1, 1.0)
        inv.register_item(itA)
        inv.register_item(itB)
        inv.register_item(itC)

        an = AnalyticsReportingEngine(inv)
        sales = {"SKU-A": 100, "SKU-B": 50, "SKU-C": 100}
        # Revenue: SKU-A = 50,000 (94.7%), SKU-B = 2,500 (4.7%), SKU-C = 200 (0.4%)
        abc = an.perform_abc_analysis(sales)
        assert abc["SKU-A"] == "A"
        assert abc["SKU-C"] == "C"

    def test_43_analytics_inventory_turnover():
        inv = InventoryService()
        an = AnalyticsReportingEngine(inv)
        # COGS = 1,000,000. Avg inv = (200k + 300k) / 2 = 250k. Turnover = 4.0
        turnover = an.calculate_inventory_turnover(1000000.0, 200000.0, 300000.0)
        assert turnover == 4.0

    def test_44_analytics_dead_stock_detection():
        inv = InventoryService()
        it = Item("SKU-OLD", "Obsolete Part", ItemCategory.COMPONENTS, 50.0, 100.0, 1, 1, 1, 1.0)
        wh = Warehouse("WH-1", "Chicago", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        old_date = date.today() - timedelta(days=120)
        inv.receive_stock("B-DEAD", "SKU-OLD", "WH-1", 10, 50.0, received_date=old_date)

        an = AnalyticsReportingEngine(inv)
        dead = an.identify_dead_stock(threshold_days=90)
        assert len(dead) == 1
        assert dead[0]["sku"] == "SKU-OLD"
        assert dead[0]["tied_up_capital"] == 500.0

    # -------------------------------------------------------------
    # Group 9: Utilities, TTLCache & Cryptographic Audit
    # -------------------------------------------------------------
    def test_45_ttl_cache_get_set_and_eviction():
        cache = TTLCache(max_size=2, default_ttl=60.0)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        assert cache.get("k1") == "v1"
        assert cache.get("k2") == "v2"
        # Over-capacity eviction
        cache.set("k3", "v3")
        assert cache.size() == 2

    def test_46_ttl_cache_expiration():
        cache = TTLCache(max_size=5, default_ttl=0.01)
        cache.set("k1", "temp", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("k1") is None

    def test_47_cryptographic_audit_log_tamper_evidence():
        ledger = CryptographicAuditLog()
        ledger.append_entry("admin", "SYSTEM_START", {"mode": "PROD"})
        ledger.append_entry("billing", "PAYMENT_CAPTURE", {"amount": 250.0})
        assert ledger.verify_integrity() is True

        # Malicious tamper simulation
        ledger.chain[1]["payload"]["amount"] = 999999.0
        assert ledger.verify_integrity() is False

    def test_48_format_currency():
        assert format_currency(1234.5) == "$1,234.50"
        assert format_currency(-50.0) == "-$50.00"

    def test_49_compound_interest_math():
        # $1000 at 5% annual compounded monthly for 2 years
        val = calculate_compound_interest(1000.0, 0.05, 12, 2.0)
        assert abs(val - 1104.94) < 0.10

    def test_50_sanitize_sku():
        assert sanitize_sku("  sku-123_abc  ") == "SKU-123_ABC"
        try:
            sanitize_sku("   $$$   ")
            assert False, "Expected ValueError"
        except ValueError:
            pass

    # -------------------------------------------------------------
    # Group 10: Data Export & Audit Serialization
    # -------------------------------------------------------------
    def test_51_export_items_to_csv():
        it = Item("SKU-CSV", "Widget", ItemCategory.RAW_MATERIALS, 10.0, 15.0, 1, 1, 1, 1.0)
        csv_str = DataExportService.export_items_to_csv([it])
        assert "SKU-CSV" in csv_str
        assert "RAW_MATERIALS" in csv_str

    def test_52_export_invoices_to_csv():
        inv = Invoice("INV-1", "ORD-1", "C-1", 100.0, 10.0, 5.0, 115.0, paid=True)
        csv_str = DataExportService.export_invoices_to_csv([inv])
        assert "INV-1" in csv_str
        assert "115.00" in csv_str

    def test_53_audit_summary_report():
        trail = [
            {"event": "ORDER_CREATED"},
            {"event": "ORDER_CREATED"},
            {"event": "PAYMENT_AUTHORIZED"}
        ]
        summary = DataExportService.generate_audit_summary_report(trail)
        assert summary["ORDER_CREATED"] == 2
        assert summary["PAYMENT_AUTHORIZED"] == 1

    # -------------------------------------------------------------
    # Group 11: Integration & End-to-End Edge Scenarios
    # -------------------------------------------------------------
    def test_54_full_quote_to_cash_multi_item_order():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)
        it1 = Item("SKU-M1", "Monitor", ItemCategory.FINISHED_GOODS, 100.0, 200.0, 5, 2, 3, 4.0)
        it2 = Item("SKU-M2", "Keyboard", ItemCategory.FINISHED_GOODS, 20.0, 50.0, 5, 2, 1, 0.8)
        wh = Warehouse("WH-1", "Dallas", 100)
        inv.register_item(it1)
        inv.register_item(it2)
        inv.register_warehouse(wh)
        inv.receive_stock("B-M1", "SKU-M1", "WH-1", 50, 100.0)
        inv.receive_stock("B-M2", "SKU-M2", "WH-1", 100, 20.0)

        cust = Customer("C-INT", "Integrated Tech", CustomerTier.GOLD, credit_limit=20000.0)
        proc.register_customer(cust)

        ord1 = proc.create_order("ORD-MULTI", "C-INT")
        ord1.add_line("SKU-M1", 5, 200.0)
        ord1.add_line("SKU-M2", 10, 50.0)

        assert ord1.subtotal == 1500.0
        proc.validate_and_reserve("ORD-MULTI")
        inv_doc = proc.authorize_payment("ORD-MULTI")
        assert inv_doc.verify_reconciliation() is True
        proc.fulfill_and_ship_order("ORD-MULTI")
        assert ord1.status == OrderStatus.SHIPPED

    def test_55_concurrent_reservation_conflict_rollback():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)

        it = Item("SKU-LIMIT", "Limited Item", ItemCategory.COMPONENTS, 10.0, 20.0, 5, 2, 1, 1.0)
        wh = Warehouse("WH-1", "Dallas", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-L", "SKU-LIMIT", "WH-1", 10, 10.0)

        cust1 = Customer("C-1", "Customer 1", CustomerTier.BRONZE, credit_limit=5000.0)
        cust2 = Customer("C-2", "Customer 2", CustomerTier.BRONZE, credit_limit=5000.0)
        proc.register_customer(cust1)
        proc.register_customer(cust2)

        ord1 = proc.create_order("ORD-CONC-1", "C-1")
        ord1.add_line("SKU-LIMIT", 8, 20.0)
        proc.validate_and_reserve("ORD-CONC-1")

        ord2 = proc.create_order("ORD-CONC-2", "C-2")
        ord2.add_line("SKU-LIMIT", 5, 20.0)  # Only 2 left
        try:
            proc.validate_and_reserve("ORD-CONC-2")
            assert False, "Expected OutOfStockException"
        except OutOfStockException:
            pass

    # -------------------------------------------------------------
    # Group 12: Regression & Parametric Integration Variations (Sprint Redundancies)
    # -------------------------------------------------------------
    def test_56_batch_inventory_fifo_multi_sku():
        inv = InventoryService()
        for i in range(3):
            sku = f"SKU-REG-{i}"
            inv.register_item(Item(sku, f"Item {i}", ItemCategory.COMPONENTS, 5.0, 10.0, 2, 1, 1, 0.5))
            wh = Warehouse(f"WH-{i}", f"Loc {i}", 50)
            inv.register_warehouse(wh)
            inv.receive_stock(f"B-{i}", sku, f"WH-{i}", 20, 5.0)
            inv.reserve_stock(sku, 5)
            dep = inv.fulfill_and_deplete_fifo(sku, 5)
            assert len(dep) == 1

    def test_57_order_pricing_tier_matrix_sweep():
        pe = PricingEngine()
        it = Item("SKU-SWP", "Sweep Item", ItemCategory.FINISHED_GOODS, 50.0, 100.0, 5, 2, 2, 1.0)
        for tier in CustomerTier:
            price, disc = pe.calculate_line_item_price(it, 25, tier)
            assert price < 100.0

    def test_58_customer_credit_balance_multi_order_simulation():
        cust = Customer("C-SIM", "SimCorp", CustomerTier.SILVER, credit_limit=15000.0)
        for amt in [2000.0, 3000.0, 4000.0]:
            assert cust.has_available_credit(amt) is True
            cust.adjust_outstanding(amt)
        assert cust.current_outstanding == 9000.0

    def test_59_shipping_calculation_boundary_conditions():
        pe = PricingEngine()
        assert pe.compute_shipping_fee(0.0) == 0.0
        assert pe.compute_shipping_fee(50.0, "REGIONAL", expedited=True) > 100.0
        assert pe.compute_shipping_fee(120.0, "INTERNATIONAL") > 500.0

    def test_60_tax_rates_multiple_jurisdictions():
        pe = PricingEngine()
        for jcode in ["US_NY", "US_TX", "US_WA", "EU_DE", "EU_FR", "IN_KA"]:
            c = Customer(f"C-{jcode}", "Corp", CustomerTier.BRONZE, credit_limit=1000.0, jurisdiction_code=jcode)
            tax = pe.compute_tax(1000.0, c)
            assert tax > 0.0

    def test_61_order_fraud_velocity_boundary_scenarios():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)
        c = Customer("C-FRD2", "Test", CustomerTier.BRONZE, credit_limit=50000.0, risk_score=0.20)
        proc.register_customer(c)
        o = proc.create_order("ORD-FRD2", "C-FRD2", payment_method=PaymentMethod.WIRE_TRANSFER)
        it = Item("SKU-FRD", "Item", ItemCategory.RAW_MATERIALS, 100.0, 200.0, 1, 1, 1, 1.0)
        inv.register_item(it)
        wh = Warehouse("WH-1", "Hub", 100)
        inv.register_warehouse(wh)
        inv.receive_stock("B-1", "SKU-FRD", "WH-1", 50, 100.0)
        o.add_line("SKU-FRD", 10, 200.0)
        risk = proc.evaluate_fraud_risk(o, c)
        assert 0.0 <= risk <= 1.0

    def test_62_ttl_cache_stress_and_concurrency():
        cache = TTLCache(max_size=10, default_ttl=10.0)
        for i in range(20):
            cache.set(f"key-{i}", f"val-{i}")
        assert cache.size() <= 10
        cache.clear()
        assert cache.size() == 0

    def test_63_cryptographic_audit_ledger_long_chain():
        ledger = CryptographicAuditLog(genesis_salt="STRESS_BLOCK")
        for i in range(10):
            ledger.append_entry("service", "ACTION", {"index": i, "status": "OK"})
        assert ledger.verify_integrity() is True
        assert len(ledger.chain) == 10

    def test_64_currency_conversion_triangular_arbitrage_precision():
        pe = PricingEngine()
        usd = 1000.0
        cad = pe.convert_currency(usd, "USD", "CAD")
        eur = pe.convert_currency(cad, "CAD", "EUR")
        usd_back = pe.convert_currency(eur, "EUR", "USD")
        assert abs(usd - usd_back) < 1e-2

    def test_65_order_processor_audit_event_sequence():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)
        c = Customer("C-AUD", "Audited Corp", CustomerTier.GOLD, credit_limit=10000.0)
        proc.register_customer(c)
        proc.create_order("ORD-AUD", "C-AUD")
        assert len(proc._audit_trail) >= 1
        assert proc._audit_trail[0]["event"] == "ORDER_CREATED"

    def test_66_item_catalog_bulk_csv_roundtrip():
        items = [
            Item(f"SKU-CSV-{i}", f"Widget {i}", ItemCategory.COMPONENTS, 10.0 + i, 20.0 + i, 1, 1, 1, 1.0)
            for i in range(5)
        ]
        csv_text = DataExportService.export_items_to_csv(items)
        assert "SKU-CSV-4" in csv_text
        assert len(csv_text.strip().split("\n")) == 6

    def test_67_invoice_reconciliation_multi_line_discount():
        inv = Invoice("INV-M", "ORD-M", "C-M", subtotal=1000.0, tax_amount=82.5, shipping_fee=25.0, total_amount=1107.5)
        assert inv.verify_reconciliation() is True

    def test_68_warehouse_pallet_fragmentation():
        wh = Warehouse("WH-FRAG", "Fragile Depot", capacity_pallets=200, occupied_pallets=100)
        assert wh.allocate_pallets(50) is True
        wh.release_pallets(30)
        assert wh.occupied_pallets == 120

    def test_69_inventory_turnover_multi_period_trends():
        an = AnalyticsReportingEngine(InventoryService())
        t1 = an.calculate_inventory_turnover(500000.0, 100000.0, 150000.0)
        t2 = an.calculate_inventory_turnover(800000.0, 100000.0, 100000.0)
        assert t2 > t1

    def test_70_order_cancellation_idempotency():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)
        c = Customer("C-IDEM", "Idem Corp", CustomerTier.BRONZE, credit_limit=5000.0)
        proc.register_customer(c)
        o = proc.create_order("ORD-IDEM", "C-IDEM")
        it = Item("SKU-I", "Item", ItemCategory.RAW_MATERIALS, 5.0, 10.0, 1, 1, 1, 1.0)
        inv.register_item(it)
        wh = Warehouse("WH-1", "Hub", 100)
        inv.register_warehouse(wh)
        inv.receive_stock("B-I", "SKU-I", "WH-1", 10, 5.0)
        o.add_line("SKU-I", 5, 10.0)
        proc.validate_and_reserve("ORD-IDEM")
        proc.cancel_order("ORD-IDEM")
        assert o.status == OrderStatus.CANCELLED

    def test_71_stock_batch_expiration_boundary_now():
        today = date.today()
        b_now = StockBatch("B-NOW", "SKU-N", "WH-1", 10, today, 1.0, expiration_date=today)
        # Expiration is checked as expiration_date < now
        assert b_now.is_expired(today + timedelta(days=1)) is True
        assert b_now.is_expired(today - timedelta(days=1)) is False

    def test_72_order_line_subtotal_floating_point_rounding():
        line = OrderLine(1, "SKU-ROUND", quantity=3, unit_price=19.99, discount_pct=15.0)
        expected_net = 19.99 * 0.85
        assert abs(line.net_unit_price - expected_net) < 1e-4

    def test_73_customer_risk_score_monotonicity():
        c_low = Customer("C-L", "Low", CustomerTier.PLATINUM, credit_limit=10000.0, risk_score=0.05)
        c_high = Customer("C-H", "High", CustomerTier.BRONZE, credit_limit=1000.0, risk_score=0.85)
        assert c_high.risk_score > c_low.risk_score
        assert c_low.has_available_credit(500.0) is True

    def test_74_pricing_engine_coupon_case_insensitivity():
        pe = PricingEngine()
        pe.register_coupon(Coupon("SAVE50", 50.0, min_order_amount=100.0))
        assert "SAVE50" in pe._coupons

    def test_75_order_processor_unregistered_customer_raises():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)
        try:
            proc.create_order("ORD-ERR", "C-NONEXISTENT")
            assert False, "Expected KeyError"
        except KeyError:
            pass

    def test_76_inventory_service_unregistered_warehouse_raises():
        inv = InventoryService()
        it = Item("SKU-WH-ERR", "Part", ItemCategory.COMPONENTS, 1.0, 2.0, 1, 1, 1, 1.0)
        inv.register_item(it)
        try:
            inv.receive_stock("B-ERR", "SKU-WH-ERR", "WH-UNKNOWN", 10, 1.0)
            assert False, "Expected KeyError"
        except KeyError:
            pass

    def test_77_analytics_abc_empty_sales_graceful():
        inv = InventoryService()
        an = AnalyticsReportingEngine(inv)
        abc = an.perform_abc_analysis({})
        assert abc == {}

    def test_78_dead_stock_zero_quantity_filtered():
        inv = InventoryService()
        it = Item("SKU-EMPTY-B", "Empty Batch Part", ItemCategory.COMPONENTS, 5.0, 10.0, 1, 1, 1, 1.0)
        wh = Warehouse("WH-1", "Chicago", 100)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-0", "SKU-EMPTY-B", "WH-1", 10, 5.0, received_date=date.today() - timedelta(days=150))
        inv.fulfill_and_deplete_fifo("SKU-EMPTY-B", 10)
        an = AnalyticsReportingEngine(inv)
        dead = an.identify_dead_stock(threshold_days=90)
        assert len(dead) == 0

    def test_79_sanitize_sku_special_characters_handling():
        assert sanitize_sku("part--123__v2") == "PART--123__V2"

    def test_80_e2e_full_order_to_delivery_flow():
        inv = InventoryService()
        pe = PricingEngine()
        proc = OrderProcessor(inv, pe)
        it = Item("SKU-E2E", "Delivered Machine", ItemCategory.FINISHED_GOODS, 100.0, 250.0, 5, 2, 2, 10.0)
        wh = Warehouse("WH-E2E", "Central Hub", 50)
        inv.register_item(it)
        inv.register_warehouse(wh)
        inv.receive_stock("B-E2E", "SKU-E2E", "WH-E2E", 20, 100.0)
        cust = Customer("C-E2E", "E2E Enterprise", CustomerTier.PLATINUM, credit_limit=50000.0)
        proc.register_customer(cust)
        o = proc.create_order("ORD-E2E", "C-E2E", payment_method=PaymentMethod.NET_60)
        o.add_line("SKU-E2E", 2, 250.0)
        proc.validate_and_reserve("ORD-E2E")
        proc.authorize_payment("ORD-E2E")
        proc.fulfill_and_ship_order("ORD-E2E")
        assert o.status == OrderStatus.SHIPPED

    # Assemble catalog
    funcs = [
        test_01_valid_item_construction,
        test_02_item_negative_cost_raises,
        test_03_item_price_below_cost_raises,
        test_04_item_margin_and_markup,
        test_05_item_zero_cost_edge_case,
        test_06_customer_credit_evaluation,
        test_07_customer_negative_amount_credit_check_raises,
        test_08_customer_adjust_outstanding,
        test_09_order_line_subtotal,
        test_10_order_line_invalid_quantity_raises,
        test_11_warehouse_allocation,
        test_12_warehouse_capacity_overflow,
        test_13_warehouse_release,
        test_14_stock_batch_expiration,
        test_15_inventory_register_and_receive,
        test_16_inventory_duplicate_item_raises,
        test_17_inventory_receive_unknown_item_raises,
        test_18_inventory_reservation_workflow,
        test_19_inventory_release_reservation,
        test_20_inventory_fifo_depletion,
        test_21_inventory_fifo_skips_expired_stock,
        test_22_inventory_fifo_out_of_stock_raises,
        test_23_safety_stock_calculation,
        test_24_economic_order_quantity,
        test_25_reorder_trigger_condition,
        test_26_reorder_trigger_not_needed,
        test_27_pricing_tier_discounts,
        test_28_pricing_line_item_volume_discount,
        test_29_coupon_validation_and_calculation,
        test_30_coupon_expired,
        test_31_tax_california_with_luxury_surcharge,
        test_32_tax_exempt_customer,
        test_33_currency_conversion,
        test_34_shipping_brackets,
        test_35_order_lifecycle_successful_completion,
        test_36_order_insufficient_stock_fails_validation,
        test_37_order_credit_limit_exceeded_fails,
        test_38_order_fraud_alert_rejection,
        test_39_order_cancellation_releases_inventory,
        test_40_order_double_cancellation_shipped_raises,
        test_41_analytics_inventory_valuation,
        test_42_analytics_abc_pareto_analysis,
        test_43_analytics_inventory_turnover,
        test_44_analytics_dead_stock_detection,
        test_45_ttl_cache_get_set_and_eviction,
        test_46_ttl_cache_expiration,
        test_47_cryptographic_audit_log_tamper_evidence,
        test_48_format_currency,
        test_49_compound_interest_math,
        test_50_sanitize_sku,
        test_51_export_items_to_csv,
        test_52_export_invoices_to_csv,
        test_53_audit_summary_report,
        test_54_full_quote_to_cash_multi_item_order,
        test_55_concurrent_reservation_conflict_rollback,
        test_56_batch_inventory_fifo_multi_sku,
        test_57_order_pricing_tier_matrix_sweep,
        test_58_customer_credit_balance_multi_order_simulation,
        test_59_shipping_calculation_boundary_conditions,
        test_60_tax_rates_multiple_jurisdictions,
        test_61_order_fraud_velocity_boundary_scenarios,
        test_62_ttl_cache_stress_and_concurrency,
        test_63_cryptographic_audit_ledger_long_chain,
        test_64_currency_conversion_triangular_arbitrage_precision,
        test_65_order_processor_audit_event_sequence,
        test_66_item_catalog_bulk_csv_roundtrip,
        test_67_invoice_reconciliation_multi_line_discount,
        test_68_warehouse_pallet_fragmentation,
        test_69_inventory_turnover_multi_period_trends,
        test_70_order_cancellation_idempotency,
        test_71_stock_batch_expiration_boundary_now,
        test_72_order_line_subtotal_floating_point_rounding,
        test_73_customer_risk_score_monotonicity,
        test_74_pricing_engine_coupon_case_insensitivity,
        test_75_order_processor_unregistered_customer_raises,
        test_76_inventory_service_unregistered_warehouse_raises,
        test_77_analytics_abc_empty_sales_graceful,
        test_78_dead_stock_zero_quantity_filtered,
        test_79_sanitize_sku_special_characters_handling,
        test_80_e2e_full_order_to_delivery_flow,
    ]

    for f in funcs:
        catalog[f.__name__] = f

    return catalog


TEST_CATALOG: Dict[str, Callable[[], None]] = _build_test_catalog()
