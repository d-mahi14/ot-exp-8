"""
TPC-H Synthetic Dataset Generator for SQLite.
Generates a representative decision-support dataset with realistic cardinalities,
foreign keys, and date ranges for TPC-H query benchmarking.
"""

import sqlite3
import random
import datetime
import os

REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]

NATIONS = [
    (0, "ALGERIA", 0), (1, "ARGENTINA", 1), (2, "BRAZIL", 1), (3, "CANADA", 1),
    (4, "EGYPT", 4), (5, "ETHIOPIA", 0), (6, "FRANCE", 3), (7, "GERMANY", 3),
    (8, "INDIA", 2), (9, "INDONESIA", 2), (10, "IRAN", 4), (11, "IRAQ", 4),
    (12, "JAPAN", 2), (13, "JORDAN", 4), (14, "KENYA", 0), (15, "MOROCCO", 0),
    (16, "MOZAMBIQUE", 0), (17, "PERU", 1), (18, "CHINA", 2), (19, "ROMANIA", 3),
    (20, "SAUDI ARABIA", 4), (21, "VIETNAM", 2), (22, "RUSSIA", 3), (23, "UNITED KINGDOM", 3),
    (24, "UNITED STATES", 1)
]

MKTSEGMENTS = ["AUTOMOBILE", "BUILDING", "CAR", "MACHINERY", "HOUSEHOLD"]
ORDER_STATUSES = ["O", "F", "P"]
ORDER_PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
RETURN_FLAGS = ["R", "A", "N"]
LINE_STATUSES = ["O", "F"]
SHIP_MODES = ["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"]


def create_tpch_schema(conn: sqlite3.Connection):
    """Creates the standard TPC-H relational schema in SQLite."""
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS lineitem;
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS customer;
    DROP TABLE IF EXISTS part;
    DROP TABLE IF EXISTS supplier;
    DROP TABLE IF EXISTS nation;
    DROP TABLE IF EXISTS region;

    CREATE TABLE region (
        r_regionkey INTEGER PRIMARY KEY,
        r_name TEXT NOT NULL,
        r_comment TEXT
    );

    CREATE TABLE nation (
        n_nationkey INTEGER PRIMARY KEY,
        n_name TEXT NOT NULL,
        n_regionkey INTEGER NOT NULL,
        n_comment TEXT,
        FOREIGN KEY (n_regionkey) REFERENCES region(r_regionkey)
    );

    CREATE TABLE supplier (
        s_suppkey INTEGER PRIMARY KEY,
        s_name TEXT NOT NULL,
        s_address TEXT,
        s_nationkey INTEGER NOT NULL,
        s_phone TEXT,
        s_acctbal REAL,
        s_comment TEXT,
        FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey)
    );

    CREATE TABLE part (
        p_partkey INTEGER PRIMARY KEY,
        p_name TEXT NOT NULL,
        p_mfgr TEXT,
        p_brand TEXT,
        p_type TEXT,
        p_size INTEGER,
        p_container TEXT,
        p_retailprice REAL,
        p_comment TEXT
    );

    CREATE TABLE customer (
        c_custkey INTEGER PRIMARY KEY,
        c_name TEXT NOT NULL,
        c_address TEXT,
        c_nationkey INTEGER NOT NULL,
        c_phone TEXT,
        c_acctbal REAL,
        c_mktsegment TEXT NOT NULL,
        c_comment TEXT,
        FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey)
    );

    CREATE TABLE orders (
        o_orderkey INTEGER PRIMARY KEY,
        o_custkey INTEGER NOT NULL,
        o_orderstatus TEXT NOT NULL,
        o_totalprice REAL,
        o_orderdate TEXT NOT NULL,
        o_orderpriority TEXT,
        o_clerk TEXT,
        o_shippriority INTEGER,
        o_comment TEXT,
        FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey)
    );

    CREATE TABLE lineitem (
        l_orderkey INTEGER NOT NULL,
        l_partkey INTEGER NOT NULL,
        l_suppkey INTEGER NOT NULL,
        l_linenumber INTEGER NOT NULL,
        l_quantity REAL NOT NULL,
        l_extendedprice REAL NOT NULL,
        l_discount REAL NOT NULL,
        l_tax REAL NOT NULL,
        l_returnflag TEXT NOT NULL,
        l_linestatus TEXT NOT NULL,
        l_shipdate TEXT NOT NULL,
        l_commitdate TEXT,
        l_receiptdate TEXT,
        l_shipinstruct TEXT,
        l_shipmode TEXT,
        l_comment TEXT,
        PRIMARY KEY (l_orderkey, l_linenumber),
        FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey),
        FOREIGN KEY (l_partkey) REFERENCES part(p_partkey),
        FOREIGN KEY (l_suppkey) REFERENCES supplier(s_suppkey)
    );
    """)
    conn.commit()


def populate_tpch_data(
    conn: sqlite3.Connection,
    n_suppliers: int = 1000,
    n_customers: int = 4000,
    n_parts: int = 4000,
    n_orders: int = 20000,
    seed: int = 42
):
    """Populates TPC-H tables with synthetic deterministic data."""
    rng = random.Random(seed)
    cur = conn.cursor()

    # 1. Regions
    region_rows = [(i, name, f"Region {name} comments") for i, name in enumerate(REGIONS)]
    cur.executemany("INSERT INTO region VALUES (?, ?, ?)", region_rows)

    # 2. Nations
    nation_rows = [(n[0], n[1], n[2], f"Nation {n[1]} comments") for n in NATIONS]
    cur.executemany("INSERT INTO nation VALUES (?, ?, ?, ?)", nation_rows)

    # 3. Suppliers
    supp_rows = []
    for s in range(1, n_suppliers + 1):
        supp_rows.append((
            s,
            f"Supplier#{s:09d}",
            f"Address {s}",
            rng.randint(0, len(NATIONS) - 1),
            f"phone-{s}",
            round(rng.uniform(-999.0, 9999.0), 2),
            f"comment {s}"
        ))
    cur.executemany("INSERT INTO supplier VALUES (?, ?, ?, ?, ?, ?, ?)", supp_rows)

    # 4. Parts
    part_rows = []
    for p in range(1, n_parts + 1):
        part_rows.append((
            p,
            f"Part {p}",
            f"Manufacturer#{rng.randint(1, 5)}",
            f"Brand#{rng.randint(1, 5)}{rng.randint(1, 5)}",
            f"Type {rng.randint(1, 10)}",
            rng.randint(1, 50),
            f"WRAP CASE {rng.randint(1, 10)}",
            round(rng.uniform(10.0, 2000.0), 2),
            f"part comment {p}"
        ))
    cur.executemany("INSERT INTO part VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", part_rows)

    # 5. Customers
    cust_rows = []
    for c in range(1, n_customers + 1):
        cust_rows.append((
            c,
            f"Customer#{c:09d}",
            f"Address {c}",
            rng.randint(0, len(NATIONS) - 1),
            f"phone-{c}",
            round(rng.uniform(-999.0, 9999.0), 2),
            rng.choice(MKTSEGMENTS),
            f"customer comment {c}"
        ))
    cur.executemany("INSERT INTO customer VALUES (?, ?, ?, ?, ?, ?, ?, ?)", cust_rows)

    # 6. Orders and Lineitems
    start_date = datetime.date(1992, 1, 1)
    date_range_days = (datetime.date(1998, 12, 31) - start_date).days

    order_rows = []
    lineitem_rows = []

    for o in range(1, n_orders + 1):
        order_days = rng.randint(0, date_range_days - 120)
        odate = start_date + datetime.timedelta(days=order_days)
        odate_str = odate.isoformat()

        custkey = rng.randint(1, n_customers)
        status = rng.choice(ORDER_STATUSES)
        priority = rng.choice(ORDER_PRIORITIES)

        # Generate 1 to 4 lineitems for each order
        n_lines = rng.randint(1, 4)
        order_total = 0.0

        for l in range(1, n_lines + 1):
            ship_delta = rng.randint(1, 90)
            sdate = odate + datetime.timedelta(days=ship_delta)
            sdate_str = sdate.isoformat()

            partkey = rng.randint(1, n_parts)
            suppkey = rng.randint(1, n_suppliers)
            qty = rng.randint(1, 50)
            base_price = rng.uniform(20.0, 500.0)
            ext_price = round(qty * base_price, 2)
            disc = round(rng.uniform(0.00, 0.10), 2)
            tax = round(rng.uniform(0.00, 0.08), 2)
            retflag = rng.choice(RETURN_FLAGS)
            lstatus = rng.choice(LINE_STATUSES)
            shipmode = rng.choice(SHIP_MODES)

            order_total += ext_price * (1.0 - disc) * (1.0 + tax)

            lineitem_rows.append((
                o, partkey, suppkey, l, float(qty), ext_price, disc, tax,
                retflag, lstatus, sdate_str, sdate_str, sdate_str,
                "DELIVER IN PERSON", shipmode, f"lineitem comment {o}-{l}"
            ))

        order_rows.append((
            o, custkey, status, round(order_total, 2), odate_str, priority,
            f"Clerk#{rng.randint(1, 100):09d}", 0, f"order comment {o}"
        ))

    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", order_rows)
    cur.executemany("INSERT INTO lineitem VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", lineitem_rows)

    conn.commit()
    print(f"[+] Populated TPC-H dataset: {len(order_rows)} orders, {len(lineitem_rows)} lineitems, {len(cust_rows)} customers.")


def build_benchmark_database(db_path: str, n_orders: int = 20000) -> sqlite3.Connection:
    """Builds and populates a new benchmark SQLite database on disk."""
    if os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    create_tpch_schema(conn)
    populate_tpch_data(conn, n_orders=n_orders)
    return conn


if __name__ == "__main__":
    db_file = os.path.join(os.path.dirname(__file__), "..", "data", "tpch_benchmark.db")
    conn = build_benchmark_database(db_file, n_orders=20000)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM lineitem")
    print(f"Total lineitems in {db_file}: {cur.fetchone()[0]}")
    conn.close()
