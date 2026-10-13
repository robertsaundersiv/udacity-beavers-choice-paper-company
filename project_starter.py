from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timedelta
from openai import OpenAI
from pathlib import Path
from rich.console import Console
from rich.markup import escape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from smolagents import ( LogLevel, ToolCallingAgent, OpenAIServerModel, tool, )
from sqlalchemy import create_engine, Engine
from sqlalchemy.sql import text
from typing import Dict, List, Any, Optional, Union, Set
import ast
import dotenv
import io
import json
import numpy as np
import os
import pandas as pd
import random
import re
import sys
import time

dotenv.load_dotenv(dotenv_path='./.env')
openai_api_key = os.getenv('UDACITY_OPENAI_API_KEY')

class StopFreeOpenAIServerModel(OpenAIServerModel):
    """OpenAIServerModel that never sends the `stop` parameter.

    Newer OpenAI models (e.g. gpt-6.1) reject `stop`, but smolagents 1.26 only
    recognizes o3/o4-mini/gpt-5 as such. Without `stop`, smolagents trims the
    output locally instead, so this is safe for every model.
    """

    @property
    def supports_stop_parameter(self) -> bool:
        return False


model = StopFreeOpenAIServerModel(
    model_id='gpt-5-mini',
    api_base='https://openai.vocareum.com/v1',
    api_key=openai_api_key,
    # Low reasoning effort keeps each request fast; tool guardrails catch errors.
    reasoning_effort='low',
)
print(openai_api_key is not None)  # Debug: Check if the API key is loaded correctly

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################
#
# Architecture (see "Agent Workflow.md"): 5 agents built with smolagents.
#   Orchestrator      -> maps items to the catalog, delegates, returns a JSON plan
#   Inventory Agent   -> stock checks and reorder decisions (JSON report)
#   Quoting Agent     -> list prices, bulk discounts, quote totals (JSON report)
#   Sales Agent       -> delivery-deadline check and recording sales (JSON report)
#   Business Advisor  -> read-only review after each reply (internal notes only)
# Every JSON report is checked against the tool records. The customer reply is
# rendered in code from the plan + recorded sales, then sanitized.
# The model is configured at the top of this file (gpt-5-mini via the proxy).


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Catalog unit_price is what the supplier charges us (stock orders are recorded
# at that price), so customers pay a list price with a markup on top.
SALES_MARKUP = 0.25
# Bulk discount tiers by total units in a quote: (minimum units, discount rate).
BULK_DISCOUNT_TIERS = [(10_000, 0.15), (5_000, 0.10), (1_000, 0.05), (0, 0.0)]
# Price floor: a discounted unit price never drops below cost x (1 + this).
MIN_MARGIN_OVER_COST = 0.05
# Run the Business Advisor after every request (it always runs at end of run).
ADVISOR_REVIEW_EVERY_REQUEST = True
ADVISOR_REPORT_PATH = "business_advisor_report.txt"
# Set AGENT_VERBOSE=1 to see smolagents' step-by-step logs instead of the
# progress display.
AGENT_VERBOSITY = LogLevel.INFO if os.getenv("AGENT_VERBOSE") == "1" else LogLevel.OFF

CATALOG: Dict[str, Dict[str, Any]] = {item["item_name"]: item for item in paper_supplies}

# Windows consoles default to cp1252; make sure status icons can be printed.
reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure):
    reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Terminal progress display (standout feature: not an agent)
# ---------------------------------------------------------------------------

class ProgressDisplay:
    """Animated terminal view of a request moving through the agent team.

    Every tool call and agent hand-off emits a status event. The latest event
    drives a live spinner, and each event is also printed as a timeline line.
    """

    STYLES = {
        "info": ("cyan", "•"),
        "handoff": ("magenta", "→"),
        "ok": ("green", "✔"),
        "fail": ("red", "✘"),
    }

    def __init__(self) -> None:
        # Bind to the real stdout so silenced helper prints never hide events.
        self.console = Console(file=sys.stdout)
        self._spinner = None

    @contextmanager
    def track(self, title: str):
        """Show a spinner for the duration of one request."""
        with self.console.status(escape(title), spinner="dots") as spinner:
            self._spinner = spinner
            try:
                yield
            finally:
                self._spinner = None

    def event(self, source: str, message: str, kind: str = "info") -> None:
        """Print one timeline line and update the spinner text."""
        color, icon = self.STYLES.get(kind, self.STYLES["info"])
        self.console.print(f"  [{color}]{icon} {escape(source)}[/{color}]: {escape(message)}")
        if self._spinner is not None:
            self._spinner.update(f"[{color}]{escape(source)}[/{color}]: {escape(message)}")


progress = ProgressDisplay()


# ---------------------------------------------------------------------------
# Shared helpers for the tools
# ---------------------------------------------------------------------------

@contextmanager
def _quiet():
    """Silence debug prints from starter helpers (get_supplier_delivery_date)."""
    with redirect_stdout(io.StringIO()):
        yield


def _validate_date(date_str: str, field_name: str = "date") -> str:
    """Return the date as YYYY-MM-DD or raise a clear error for the agent."""
    try:
        return datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format, got {date_str!r}")


def _stocked_items() -> Dict[str, int]:
    """Items the company stocks (inventory table) mapped to their min_stock_level."""
    inventory_df = pd.read_sql("SELECT item_name, min_stock_level FROM inventory", db_engine)
    return dict(zip(inventory_df["item_name"], inventory_df["min_stock_level"].astype(int)))


def _stock_on(item_name: str, as_of_date: str) -> int:
    """Net units of one item on a date, via the get_stock_level helper."""
    stock_df = get_stock_level(item_name, as_of_date)
    return int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0


def _list_price(item_name: str) -> float:
    """Customer list price per unit: supplier cost plus the sales markup."""
    return round(CATALOG[item_name]["unit_price"] * (1 + SALES_MARKUP), 3)


def _bulk_discount_rate(total_units: int) -> float:
    """Discount rate for the tier that total_units falls into."""
    for min_units, rate in BULK_DISCOUNT_TIERS:
        if total_units >= min_units:
            return rate
    return 0.0


def _to_json(data: Any) -> str:
    """Serialize tool output for the LLM (handles numpy and date values)."""
    return json.dumps(data, default=str)


_catalog_names = list(CATALOG)
_catalog_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
_catalog_matrix = _catalog_vectorizer.fit_transform(_catalog_names)


def _closest_catalog_names(description: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Top catalog names for a phrase by TF-IDF character n-gram similarity."""
    scores = cosine_similarity(_catalog_vectorizer.transform([description]), _catalog_matrix)[0]
    best = scores.argsort()[::-1][:top_n]
    return [{"item_name": _catalog_names[i], "score": round(float(scores[i]), 2)} for i in best]


# Paper sizes other than A4. The catalog's generic papers ("Glossy paper") are
# standard A4/letter, so an A3 or A5 request is a different product.
_NON_A4_PAPER_SIZE = re.compile(r"\bA(?:10|[0-35-9])\b", re.IGNORECASE)


def _size_mismatch(description: str, item_name: str) -> Optional[str]:
    """The paper size asked for (e.g. 'A3') if the catalog item is not that size, else None."""
    for size in _NON_A4_PAPER_SIZE.findall(description):
        if not re.search(rf"\b{size}\b", item_name, re.IGNORECASE):
            return size.upper()
    return None


def _require_catalog_name(item_name: str) -> None:
    """Reject anything that is not an exact catalog name, suggesting close ones."""
    if item_name not in CATALOG:
        suggestions = [c["item_name"] for c in _closest_catalog_names(item_name)]
        raise ValueError(
            f"'{item_name}' is not an exact catalog item name. Closest exact names: {suggestions}. "
            "Retry with one of these if it is the same product; otherwise the item is not in our catalog."
        )


# ---------------------------------------------------------------------------
# Per-request context
# ---------------------------------------------------------------------------
# Dates are parsed from the request text in code, not passed around by the
# LLM: in testing, agents dropped or invented dates (e.g. 2023-10-03), which
# made stock and cash read as zero. Tools that read or write the database use
# the dates held here. The context also makes writes idempotent, so a repeated
# delegation cannot record the same sale or customer reorder twice.

_MONTH_DATE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})"
)
_request_context: Dict[str, Any] = {}


def _extract_request_date(request_text: str) -> str:
    """Pull the request date the harness appends to each request."""
    match = re.search(r"Date of request: (\d{4}-\d{2}-\d{2})", request_text)
    return match.group(1) if match else datetime.now().strftime("%Y-%m-%d")


def _extract_deliver_by(request_text: str, request_date: str) -> str:
    """Latest 'Month D, YYYY' date in the request, else request date + 14 days."""
    body = request_text.split("(Date of request:")[0]
    dates = [datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
             for month, day, year in _MONTH_DATE.findall(body)]
    if dates:
        return max(dates).strftime("%Y-%m-%d")
    return (datetime.strptime(request_date, "%Y-%m-%d") + timedelta(days=14)).strftime("%Y-%m-%d")


def start_request_context(request_date: str, deliver_by: str) -> None:
    """Open the context for one customer request (dates + idempotency logs)."""
    _request_context.clear()
    _request_context.update({
        "request_date": _validate_date(request_date, "request_date"),
        "deliver_by": _validate_date(deliver_by, "deliver_by"),
        "item_status": {},      # item_name -> AVAILABLE / RESTOCK_NEEDED / NOT_STOCKED / UNAVAILABLE
        "order_reorders": {},   # item_name -> reorder result
        "sold_lines": {},       # (item_name, quantity) -> {transaction_id, line_total, delivery}
        "rejected_lines": {},   # item_name -> reason code from the Sales Agent's report
        "quote_lines": {},      # item_name -> latest quote line from calculate_quote_tool
        "discount_rate": 0.0,   # bulk discount rate of the latest quote
        "quote_rationale": "",  # customer-friendly discount explanation of the latest quote
        "agents_called": set(), # names of worker agents the orchestrator delegated to
    })


def _current(key: str) -> Any:
    """Read a value from the active request context."""
    if key not in _request_context:
        raise RuntimeError("No active request: call start_request_context() first.")
    return _request_context[key]


# ---------------------------------------------------------------------------
# Tools for the orchestrator
# ---------------------------------------------------------------------------


@tool
def match_catalog_items_tool(item_descriptions: List[str]) -> str:
    """
    Map the customer's wording to exact catalog item names.

    Uses TF-IDF character n-gram similarity over the paper_supplies catalog
    (no database helper). Returns the 3 closest catalog names per description
    so you can pick the one that is genuinely the same product. Candidates of
    the wrong paper size (e.g. "Glossy paper" for "A3 glossy paper") are
    removed, because we only carry standard sizes.

    Args:
        item_descriptions: Item phrases as the customer wrote them, e.g. ["A4 glossy paper", "heavy cardstock (white)"].

    Returns:
        JSON list with up to 3 candidate item names and similarity scores (0-1) per description, plus a note when a size is not carried.
    """
    progress.event("Orchestrator", f"matching {len(item_descriptions)} item(s) to the catalog")
    matches = []
    for description in item_descriptions:
        candidates = _closest_catalog_names(description)
        sized = [c for c in candidates if not _size_mismatch(description, c["item_name"])]
        match = {"description": description, "candidates": sized}
        if len(sized) < len(candidates):
            size = _size_mismatch(description, "")  # the non-A4 size the customer asked for
            match["note"] = (f"We do not carry {size} size; a candidate of a different size is not the "
                             "same product." + (" This item is NOT IN OUR CATALOG." if not sized else ""))
        matches.append(match)
    return _to_json(matches)


# ---------------------------------------------------------------------------
# Tools for inventory agent
# ---------------------------------------------------------------------------

@tool
def check_stock_tool(item_name: str, quantity_requested: int) -> str:
    """
    Check one item's stock against a requested quantity and its reorder level.

    Uses get_stock_level() on the request date plus min_stock_level from the
    inventory table.

    Args:
        item_name: Exact catalog item name.
        quantity_requested: Units the customer wants.

    Returns:
        JSON with status (AVAILABLE, RESTOCK_NEEDED or NOT_STOCKED), current_stock, min_stock_level, needs_reorder and suggested_reorder_quantity.
    """
    _require_catalog_name(item_name)
    request_date = _current("request_date")
    progress.event("Inventory Agent", f"checking stock of {item_name} x{quantity_requested:,}")
    stocked = _stocked_items()
    if item_name not in stocked:
        _current("item_status")[item_name] = "NOT_STOCKED"
        return _to_json({"item_name": item_name, "status": "NOT_STOCKED",
                         "note": "The company does not stock this item, so it cannot be supplied."})

    current_stock = _stock_on(item_name, request_date)
    min_level = stocked[item_name]
    needs_reorder = current_stock - quantity_requested < min_level
    status = "AVAILABLE" if current_stock >= quantity_requested else "RESTOCK_NEEDED"
    _current("item_status")[item_name] = status
    return _to_json({
        "item_name": item_name,
        "status": status,
        "current_stock": current_stock,
        "min_stock_level": min_level,
        "needs_reorder": needs_reorder,
        "suggested_reorder_quantity": max(0, quantity_requested + min_level - current_stock),
        "request_date": request_date,
    })


@tool
def inventory_snapshot_tool() -> str:
    """
    List every item in stock on the request date, with units on hand.

    Uses get_all_inventory(). Useful for general "what do you have?" questions
    and as a health check before large orders.

    Returns:
        JSON mapping item names to units in stock (items with zero stock are omitted).
    """
    request_date = _current("request_date")
    progress.event("Inventory Agent", "taking an inventory snapshot")
    snapshot = get_all_inventory(request_date)
    return _to_json({"as_of_date": request_date,
                     "items_in_stock": {name: int(units) for name, units in snapshot.items()}})


@tool
def check_cash_tool() -> str:
    """
    Check the company's cash balance on the request date before buying stock.

    Uses get_cash_balance().

    Returns:
        JSON with the cash balance in dollars.
    """
    request_date = _current("request_date")
    progress.event("Inventory Agent", "checking cash available for reorders")
    return _to_json({"as_of_date": request_date, "cash_balance": round(get_cash_balance(request_date), 2)})


@tool
def supplier_lead_time_tool(quantity: int) -> str:
    """
    Estimate when a supplier restock ordered on the request date would arrive.

    Uses get_supplier_delivery_date(): up to 10 units same day, 11-100 units
    1 day, 101-1000 units 4 days, over 1000 units 7 days.

    Args:
        quantity: Units to order from the supplier.

    Returns:
        JSON with the expected arrival date and whether it meets the customer's deliver-by date.
    """
    request_date, deliver_by = _current("request_date"), _current("deliver_by")
    progress.event("Inventory Agent", f"asking supplier lead time for {quantity:,} units")
    with _quiet():
        arrival = get_supplier_delivery_date(request_date, quantity)
    return _to_json({"quantity": quantity, "expected_arrival": arrival, "deliver_by": deliver_by,
                     "arrives_in_time": arrival <= deliver_by})


@tool
def reorder_stock_tool(item_name: str, quantity: int, for_customer_order: bool) -> str:
    """
    Buy stock of an item the company stocks from the supplier, on the request date.

    Records a 'stock_orders' transaction with create_transaction() at
    quantity x unit_price. Refuses items the company does not stock, purchases
    larger than the cash balance and, for customer orders, restocks that would
    arrive after the customer's deliver-by date.

    Args:
        item_name: Exact catalog name of a stocked item.
        quantity: Units to buy (use suggested_reorder_quantity from check_stock_tool).
        for_customer_order: true if the restock is needed to fill this customer's order (status RESTOCK_NEEDED); false for a routine top-up to the minimum level.

    Returns:
        JSON with ok, and either the transaction id, cost and arrival date, or the reason it was refused.
    """
    _require_catalog_name(item_name)
    request_date, deliver_by = _current("request_date"), _current("deliver_by")
    if item_name not in _stocked_items():
        return _to_json({"ok": False, "reason": f"{item_name} is not an item the company stocks."})
    if quantity <= 0:
        return _to_json({"ok": False, "reason": "quantity must be positive."})
    already = _current("order_reorders").get(item_name)
    if already:
        return _to_json({**already, "note": "Already reordered for this request; nothing new recorded."})

    cost = round(quantity * CATALOG[item_name]["unit_price"], 2)
    with _quiet():
        arrival = get_supplier_delivery_date(request_date, quantity)
    if for_customer_order and arrival > deliver_by:
        _current("item_status")[item_name] = "UNAVAILABLE"
        progress.event("Inventory Agent", f"restock of {item_name} would arrive {arrival}, too late", "fail")
        return _to_json({"ok": False, "reason": f"Restock would arrive {arrival}, after the deliver-by date {deliver_by}."})
    if cost > get_cash_balance(request_date):
        if for_customer_order:
            _current("item_status")[item_name] = "UNAVAILABLE"
        progress.event("Inventory Agent", f"not enough cash to restock {item_name}", "fail")
        return _to_json({"ok": False, "reason": "Insufficient funds for this purchase."})

    transaction_id = create_transaction(item_name, "stock_orders", quantity, cost, request_date)
    result = {"ok": True, "transaction_id": transaction_id, "item_name": item_name,
              "quantity": quantity, "cost": cost, "expected_arrival": arrival}
    _current("order_reorders")[item_name] = result
    if for_customer_order:
        _current("item_status")[item_name] = "AVAILABLE"
    progress.event("Inventory Agent", f"reordered {quantity:,} x {item_name} (arrives {arrival})", "ok")
    return _to_json(result)


# ---------------------------------------------------------------------------
# Tools for quoting agent
# ---------------------------------------------------------------------------

@tool
def quote_history_tool(search_terms: List[str]) -> str:
    """
    Find similar past quotes to guide pricing and the explanation style.

    Uses search_quote_history(). All terms must match, so pass 1-2 short
    keywords, e.g. ["cardstock"] or ["ceremony"].

    Args:
        search_terms: 1-2 short keywords (an item word or the event type).

    Returns:
        JSON list of up to 5 past quotes with total amount, order size, event type and explanation.
    """
    terms = [term.strip() for term in search_terms if term and term.strip()][:2]
    progress.event("Quoting Agent", f"searching quote history for {terms}")
    history = search_quote_history(terms, limit=5)
    if not history and len(terms) > 1:
        history = search_quote_history(terms[:1], limit=5)
    return _to_json([
        {
            "total_amount": row["total_amount"],
            "job_type": row["job_type"],
            "order_size": row["order_size"],
            "event_type": row["event_type"],
            "quote_explanation": str(row["quote_explanation"])[:300],
        }
        for row in history
    ])


@tool
def price_lookup_tool(item_names: List[str]) -> str:
    """
    Look up the customer list price per unit for catalog items.

    Reads the paper_supplies catalog (no database helper). Use it for price
    questions without quantities; use calculate_quote_tool for full quotes.

    Args:
        item_names: Exact catalog item names.

    Returns:
        JSON mapping each item name to its list price per unit, or to the closest exact names if it is not a catalog name.
    """
    progress.event("Quoting Agent", f"looking up list prices for {len(item_names)} item(s)")
    return _to_json({
        name: (_list_price(name) if name in CATALOG
               else {"not_in_catalog": True, "closest": [c["item_name"] for c in _closest_catalog_names(name)]})
        for name in item_names
    })


@tool
def calculate_quote_tool(item_names: List[str], quantities: List[int]) -> str:
    """
    Price a set of items: line totals, bulk discount tier and quote total.

    The discount tier is based on total units across the quote: 5% from 1,000
    units, 10% from 5,000 and 15% from 10,000. Prices never drop below the
    company's minimum price. Pure Python (no database helper).

    Args:
        item_names: Exact catalog item names to quote.
        quantities: Units for each item, in the same order as item_names.

    Returns:
        JSON with one line per item (unit_price, subtotal, discount, line_total), the discount rate, the quote total and a customer-friendly rationale.
    """
    if len(item_names) != len(quantities):
        raise ValueError("item_names and quantities must have the same length.")
    for name in item_names:
        _require_catalog_name(name)

    total_units = int(sum(quantities))
    rate = _bulk_discount_rate(total_units)
    progress.event("Quoting Agent", f"pricing {total_units:,} units, bulk discount {rate:.0%}")

    lines = []
    for name, quantity in zip(item_names, quantities):
        unit_price = _list_price(name)
        subtotal = round(unit_price * quantity, 2)
        price_floor = CATALOG[name]["unit_price"] * (1 + MIN_MARGIN_OVER_COST) * quantity
        line_total = round(max(subtotal * (1 - rate), price_floor), 2)
        lines.append({
            "item_name": name,
            "quantity": int(quantity),
            "unit_price": unit_price,
            "subtotal": subtotal,
            "discount": round(subtotal - line_total, 2),
            "line_total": line_total,
        })

    total = round(sum(line["line_total"] for line in lines), 2)
    if rate > 0:
        rationale = (f"Your order totals {total_units:,} units, which qualifies for our "
                     f"{rate:.0%} bulk discount on every item.")
    else:
        rationale = (f"Bulk discounts start at 1,000 units (5%), rising to 10% at 5,000 and "
                     f"15% at 10,000; this order has {total_units:,} units, so list prices apply.")
    # Keep the quote so the Quoting Agent's report and the customer reply can be checked against it.
    if "quote_lines" in _request_context:
        _request_context["quote_lines"].update({line["item_name"]: line for line in lines})
        _request_context.update(discount_rate=rate, quote_rationale=rationale)
    return _to_json({"lines": lines, "total_units": total_units, "discount_rate": rate,
                     "quote_total": total, "rationale": rationale})


# ---------------------------------------------------------------------------
# Tools for ordering (sales) agent
# ---------------------------------------------------------------------------

@tool
def delivery_check_tool(quantity: int) -> str:
    """
    Estimate delivery for one order line and check it meets the customer's deadline.

    Uses get_supplier_delivery_date() from the request date as the delivery
    estimate for the line, compared with the customer's deliver-by date.

    Args:
        quantity: Units on this order line.

    Returns:
        JSON with estimated_delivery, deliver_by and meets_deadline (true/false).
    """
    request_date, deliver_by = _current("request_date"), _current("deliver_by")
    with _quiet():
        estimated_delivery = get_supplier_delivery_date(request_date, quantity)
    meets_deadline = estimated_delivery <= deliver_by
    progress.event("Sales Agent", f"{quantity:,} units deliverable {estimated_delivery} "
                                  f"(deadline {deliver_by})", "ok" if meets_deadline else "fail")
    return _to_json({"estimated_delivery": estimated_delivery, "deliver_by": deliver_by,
                     "meets_deadline": meets_deadline})


@tool
def record_sale_tool(item_name: str, quantity: int, line_total: float) -> str:
    """
    Record a confirmed sale for one order line on the request date.

    Refuses lines that cannot be delivered by the deliver-by date
    (get_supplier_delivery_date()), that are short of stock (re-checked with
    get_stock_level()) or that are priced below the company's minimum;
    otherwise writes a 'sales' transaction with create_transaction().

    Args:
        item_name: Exact catalog item name.
        quantity: Units sold.
        line_total: Quoted total for this line in dollars (from the quote).

    Returns:
        JSON with ok, and either the transaction id and delivery date or the reason the sale was refused.
    """
    _require_catalog_name(item_name)
    request_date, deliver_by = _current("request_date"), _current("deliver_by")
    sold_lines = _current("sold_lines")
    if (item_name, quantity) in sold_lines:
        return _to_json({"ok": True, **sold_lines[(item_name, quantity)],
                         "note": "Already recorded for this request; nothing new recorded."})

    # Guard: never record a line the customer cannot receive in time.
    with _quiet():
        delivery = get_supplier_delivery_date(request_date, quantity)
    if delivery > deliver_by:
        progress.event("Sales Agent", f"refused {item_name}: delivery {delivery} misses {deliver_by}", "fail")
        return _to_json({"ok": False, "reason": f"Cannot be delivered by {deliver_by}; earliest delivery {delivery}."})

    in_stock = _stock_on(item_name, request_date)
    if in_stock < quantity:
        progress.event("Sales Agent", f"insufficient stock of {item_name}", "fail")
        return _to_json({"ok": False, "reason": "Insufficient stock to fulfil this line."})
    price_floor = round(CATALOG[item_name]["unit_price"] * (1 + MIN_MARGIN_OVER_COST) * quantity, 2)
    if line_total < price_floor:
        return _to_json({"ok": False, "reason": "Price is below the minimum allowed; use the quoted line_total."})

    transaction_id = create_transaction(item_name, "sales", quantity, round(line_total, 2), request_date)
    sold_lines[(item_name, quantity)] = {"transaction_id": transaction_id,
                                         "line_total": round(line_total, 2), "delivery": delivery}
    progress.event("Sales Agent", f"sold {quantity:,} x {item_name} for ${line_total:,.2f}", "ok")
    return _to_json({"ok": True, "transaction_id": transaction_id, "item_name": item_name,
                     "quantity": quantity, "line_total": round(line_total, 2), "delivery": delivery})


# ---------------------------------------------------------------------------
# Tools for business advisor agent (read-only)
# ---------------------------------------------------------------------------

@tool
def financial_report_tool(as_of_date: str) -> str:
    """
    Summarize the company's finances on a date.

    Uses generate_financial_report().

    Args:
        as_of_date: Report date, YYYY-MM-DD.

    Returns:
        JSON with cash balance, inventory value, total assets and the top-selling products.
    """
    as_of_date = _validate_date(as_of_date, "as_of_date")
    progress.event("Business Advisor", "reading the financial report")
    report = generate_financial_report(as_of_date)
    # Drop the seeded starting-cash row (a 'sales' entry with no item name).
    top_sellers = [
        {"item_name": row["item_name"], "units": int(row["total_units"]),
         "revenue": round(float(row["total_revenue"]), 2)}
        for row in report["top_selling_products"]
        if isinstance(row["item_name"], str)
    ]
    return _to_json({
        "as_of_date": as_of_date,
        "cash_balance": round(report["cash_balance"], 2),
        "inventory_value": round(report["inventory_value"], 2),
        "total_assets": round(report["total_assets"], 2),
        "top_selling_products": top_sellers,
    })


@tool
def cash_trend_tool(before_date: str, after_date: str) -> str:
    """
    Compare the cash balance on two dates.

    Uses get_cash_balance() for each date.

    Args:
        before_date: Earlier date, YYYY-MM-DD (e.g. the day before a request).
        after_date: Later date, YYYY-MM-DD (e.g. the request date).

    Returns:
        JSON with both balances and the change.
    """
    before_date = _validate_date(before_date, "before_date")
    after_date = _validate_date(after_date, "after_date")
    progress.event("Business Advisor", f"comparing cash {before_date} -> {after_date}")
    before, after = get_cash_balance(before_date), get_cash_balance(after_date)
    return _to_json({"before": round(before, 2), "after": round(after, 2), "change": round(after - before, 2)})


@tool
def stock_health_tool(as_of_date: str) -> str:
    """
    Find stocked items that are low or out of stock.

    Uses get_all_inventory() plus min_stock_level from the inventory table.

    Args:
        as_of_date: Date to check, YYYY-MM-DD.

    Returns:
        JSON listing items below their minimum level and items out of stock.
    """
    as_of_date = _validate_date(as_of_date, "as_of_date")
    progress.event("Business Advisor", "checking stock health")
    snapshot = get_all_inventory(as_of_date)
    below_min, out_of_stock = [], []
    for item_name, min_level in _stocked_items().items():
        units = int(snapshot.get(item_name, 0))
        if units == 0:
            out_of_stock.append(item_name)
        elif units < min_level:
            below_min.append({"item_name": item_name, "stock": units, "min_stock_level": min_level})
    return _to_json({"as_of_date": as_of_date, "below_min_level": below_min, "out_of_stock": out_of_stock})


# ---------------------------------------------------------------------------
# Agent instructions
# ---------------------------------------------------------------------------

INVENTORY_INSTRUCTIONS = """
You are the Inventory Agent of Beaver's Choice Paper Company. You decide whether
each requested item can be supplied and you reorder stock from the supplier.
The request date and the customer's deliver-by date are applied by the tools
automatically; you never pass dates.

For EACH item in the task (exact catalog names are given):
1. Call check_stock_tool with the item name and quantity.
2. status NOT_STOCKED: the item is UNAVAILABLE (we do not stock it).
3. status AVAILABLE: the item is AVAILABLE. If needs_reorder is true, also top
   up stock with reorder_stock_tool(item, suggested_reorder_quantity,
   for_customer_order=false). The item stays AVAILABLE even if that is refused.
4. status RESTOCK_NEEDED: call supplier_lead_time_tool(suggested_reorder_quantity)
   and check_cash_tool(). If arrives_in_time is true and cash covers the
   purchase, call reorder_stock_tool(item, suggested_reorder_quantity,
   for_customer_order=true). If ok is true the item is AVAILABLE (restocked);
   otherwise it is UNAVAILABLE with the reason (cannot be restocked by the
   customer's date, or not enough funds).
For large orders (5,000+ total units) call inventory_snapshot_tool once first
as a stock health check.
If a tool says a name is not an exact catalog name, retry with the closest
exact name only if it is clearly the same product.

Final answer: ONLY this JSON object, no other text:
{"items": [{"item_name": "<exact name>", "quantity": <int>,
            "status": "AVAILABLE" or "UNAVAILABLE",
            "reason": null or "not_stocked" or "cannot_supply_in_time"}]}
reason is null for AVAILABLE items, "not_stocked" for NOT_STOCKED items and
"cannot_supply_in_time" when a needed restock is too late or not affordable.
Add no other fields: never stock counts, costs, cash or supplier dates.
""".strip()

QUOTING_INSTRUCTIONS = """
You are the Quoting Agent of Beaver's Choice Paper Company. You price orders.

1. Call quote_history_tool once with 1-2 keywords (an item word such as
   "cardstock" or the event type such as "ceremony") to see how similar orders
   were priced and explained.
2. Call calculate_quote_tool once with ALL item names and quantities. Never
   compute or change prices yourself; use its numbers exactly.
   (Use price_lookup_tool only for a price question without quantities.)

Final answer: ONLY this JSON object, copying calculate_quote_tool's numbers
exactly:
{"lines": [{"item_name": "<exact name>", "quantity": <int>,
            "unit_price": <number>, "line_total": <number>}],
 "discount_rate": <number>, "quote_total": <number>,
 "rationale": "<the tool's rationale>"}
You may add to the rationale that the price is in line with similar past
orders. Never mention cost, markup or margin.
""".strip()

SALES_INSTRUCTIONS = """
You are the Sales Agent of Beaver's Choice Paper Company. You finalize orders.
The request date and the customer's deliver-by date are applied by the tools
automatically; you never pass dates.

For EACH quote line (item name, quantity, line total):
1. Call delivery_check_tool with the quantity.
2. If meets_deadline is false: the line is REJECTED - it cannot be delivered
   by the deliver-by date (give the estimated delivery date). Do not record it.
3. Otherwise call record_sale_tool with the item name, quantity and the quoted
   line total. If ok is true the line is SOLD; otherwise it is REJECTED with
   the reason returned.

Final answer: ONLY this JSON object, no other text:
{"lines": [{"item_name": "<exact name>", "quantity": <int>,
            "status": "SOLD" or "REJECTED", "line_total": <number>,
            "delivery_date": "YYYY-MM-DD" or null,
            "reason": null or "cannot_deliver_by_deadline" or "not_available"}],
 "total_charged": <sum of SOLD line totals>}
SOLD lines use the delivery date returned by record_sale_tool and reason null.
REJECTED lines use delivery_date null and reason "cannot_deliver_by_deadline"
(too late) or "not_available" (any other refusal).
""".strip()

ORCHESTRATOR_INSTRUCTIONS = """
You are the Orchestrator (customer service lead) of Beaver's Choice Paper
Company. You never check stock, set prices or record sales yourself: you
delegate to your team and decide the outcome of every requested item. The
request date and deliver-by date are given at the top of each task and are
applied by the team's tools automatically. Each team member reports back as a
JSON object.

For every request:
1. List every item the customer wants with its quantity (use the number the
   customer gives; do not convert reams, packs or rolls).
2. Call match_catalog_items_tool with all item phrases at once. For each item
   choose the candidate that is genuinely the same product (e.g. "A4 glossy
   paper" -> "Glossy paper", "heavy cardstock" -> "Cardstock", "A4 printer
   paper" -> "A4 paper"). If no candidate is the same product (e.g. A3 or A5
   paper, balloons, tickets, cardboard), or the tool returns no candidates,
   the item is NOT IN OUR CATALOG and is not sent to the team. Never use a
   catalog name the tool did not return for that item.
3. Call inventory_agent ONCE with every matched item as a line
   "<exact item name> | <quantity>". Copy the exact item_name strings from
   match_catalog_items_tool character for character - never the customer's
   wording.
4. If at least one item has status AVAILABLE, call quoting_agent ONCE with
   only the AVAILABLE items as "<exact item name> | <quantity>" (and the event
   type, if mentioned). Skip steps 4-5 if nothing is AVAILABLE.
5. Call sales_agent ONCE with every quote line as
   "<exact item name> | <quantity> | <line_total>".
6. Final answer: ONLY this JSON object, no other text. The customer reply
   (prices, discount, total, delivery dates and reasons) is written from it in
   code using the recorded sales, so do not write the reply yourself:
   {"items": [{"requested": "<customer's item wording, without the quantity>",
               "quantity": <int>,
               "catalog_item": "<exact item name>" or null,
               "outcome": "SOLD" or "NOT_SUPPLIED",
               "reason": null or "not_in_catalog" or "not_stocked" or
                         "cannot_supply_in_time" or
                         "cannot_deliver_by_deadline" or "not_available"}],
    "note": "<one or two short, friendly sentences to the customer>"}
   - One entry per item the customer asked for, in their order.
     catalog_item is null (reason "not_in_catalog") for items NOT IN OUR
     CATALOG.
   - outcome SOLD (reason null) only for lines sales_agent reported SOLD;
     every other item is NOT_SUPPLIED with the reason from the team's report.
   - note: a warm, personal touch, e.g. wishing their event well, or an
     apology if nothing could be supplied. No prices, numbers, dates or item
     lists, no internal information (costs, stock counts, suppliers, agent or
     tool names, errors) and no offers of anything the company did not do
     (substitutions, sourcing, expedited delivery, invoices, tracking, or
     "if you'd like, I can...").
""".strip()

ADVISOR_INSTRUCTIONS = """
You are the Business Advisor of Beaver's Choice Paper Company. You are
read-only and internal: your notes go to management, never to customers.

Given a request outcome (or an end-of-run review), call financial_report_tool
and stock_health_tool for the review date, and cash_trend_tool with
before_date = the day before the review date and after_date = the review date
(for an end-of-run review use the first and last dates given). Then give a
one-line summary of the numbers and 2-4 short, specific recommendations as
bullet points to improve revenue or efficiency (e.g. items worth adding to the
range, reorder levels to raise, discount tiers, recurring reasons for lost
sales). Base each on the numbers or the outcome you saw. A structured
lost-demand log (unsold items with reason codes, request and unit counts) is
included: base any recommendation about the product range or stock levels on
it and cite its counts. Keep it under 150 words. This is a written log: do not
ask questions or offer further actions.
""".strip()


# ---------------------------------------------------------------------------
# Set up your agents and create an orchestration agent that will manage them.
# ---------------------------------------------------------------------------

class WorkerAgent(ToolCallingAgent):
    """ToolCallingAgent that reports hand-offs to the progress display."""

    def __init__(self, *args, display_name: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.display_name = display_name

    def __call__(self, task: str, **kwargs):
        progress.event("Orchestrator", f"delegating to {self.display_name}", "handoff")
        if "agents_called" in _request_context:
            _request_context["agents_called"].add(self.name)
        report = super().__call__(task, **kwargs)
        progress.event(self.display_name, "reported back to Orchestrator", "ok")
        return report


# ---------------------------------------------------------------------------
# Structured hand-offs: every agent reports back as a fixed JSON object
# ---------------------------------------------------------------------------
# Worker reports carry only what the next agent needs (status, reason code,
# price, delivery date) - never stock counts, costs or cash - so internal data
# cannot travel up to the customer reply. Each report is checked against what
# the tools actually recorded; a mismatch is sent back to the agent to fix.

INVENTORY_REASONS = {"not_stocked", "cannot_supply_in_time"}
SALES_REASONS = {"cannot_deliver_by_deadline", "not_available"}
PLAN_REASONS = {"not_in_catalog"} | INVENTORY_REASONS | SALES_REASONS


def _parse_json_answer(answer: Any) -> Dict[str, Any]:
    """Read an agent's final answer as a JSON object (a dict passes through)."""
    if isinstance(answer, dict):
        return answer
    answer_text = str(answer)
    start, end = answer_text.find("{"), answer_text.rfind("}")
    parsed = None
    if start != -1 and end > start:
        body = answer_text[start:end + 1]
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            # The model sometimes passes a Python dict literal (single quotes, None).
            try:
                parsed = ast.literal_eval(body)
            except (ValueError, SyntaxError):
                parsed = None
    if not isinstance(parsed, dict):
        raise ValueError("The final answer must be ONLY the JSON object described in your instructions.")
    return parsed


def _check_keys(obj: Any, keys: Set[str], label: str) -> Dict[str, Any]:
    """Require a JSON object with exactly these keys (no extra fields)."""
    if not isinstance(obj, dict) or set(obj) != keys:
        found = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        raise ValueError(f"{label} must have exactly the keys {sorted(keys)}, got {found}. "
                         "Add no other fields (no stock counts, costs, cash or supplier details).")
    return obj


def _check_rows(rows: Any, keys: Set[str], label: str) -> List[Dict[str, Any]]:
    """Require a non-empty list of JSON objects, each with exactly these keys."""
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"'{label}' must be a non-empty JSON list.")
    return [_check_keys(row, keys, f"Each entry in '{label}'") for row in rows]


def _as_int(value: Any, label: str) -> int:
    """A whole-number field from a JSON report."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number, got {value!r}.")


def _same_money(value: Any, expected: float) -> bool:
    """True if a reported dollar amount equals the recorded one (to the cent)."""
    try:
        return abs(float(value) - expected) < 0.005
    except (TypeError, ValueError):
        return False


def _reject(source: str, message: str) -> None:
    """Show a rejected report on the progress display, then fail the check."""
    progress.event(source, "report did not match the records - redoing it", "fail")
    raise ValueError(message)


def _inventory_report_problem(report: Dict[str, Any]) -> Optional[str]:
    """Why an Inventory Agent report disagrees with the stock checks, or None."""
    item_status = _current("item_status")
    for row in _check_rows(_check_keys(report, {"items"}, "The report")["items"],
                           {"item_name", "quantity", "status", "reason"}, "items"):
        name, recorded = row["item_name"], item_status.get(row["item_name"])
        if recorded is None:
            return f"'{name}' was never checked: call check_stock_tool for it first."
        status = "AVAILABLE" if recorded == "AVAILABLE" else "UNAVAILABLE"
        reason = None if status == "AVAILABLE" else (
            "not_stocked" if recorded == "NOT_STOCKED" else "cannot_supply_in_time")
        if (row["status"], row["reason"]) != (status, reason):
            return f"'{name}' must be reported with status {status!r} and reason {reason!r}."
    return None


def _quote_report_problem(report: Dict[str, Any]) -> Optional[str]:
    """Why a Quoting Agent report disagrees with calculate_quote_tool, or None."""
    report = _check_keys(report, {"lines", "discount_rate", "quote_total", "rationale"}, "The report")
    quoted = _current("quote_lines")
    for row in _check_rows(report["lines"], {"item_name", "quantity", "unit_price", "line_total"}, "lines"):
        line = quoted.get(row["item_name"])
        if (line is None or _as_int(row["quantity"], "quantity") != line["quantity"]
                or not _same_money(row["line_total"], line["line_total"])):
            return (f"Line {row} does not match calculate_quote_tool. Call it, then copy its "
                    "item_name, quantity, unit_price and line_total exactly.")
    if not _same_money(report["quote_total"], sum(float(row["line_total"]) for row in report["lines"])):
        return "quote_total must equal the sum of the line totals."
    if not str(report["rationale"]).strip():
        return "Give the customer-friendly rationale from calculate_quote_tool."
    return None


def _sales_report_problem(report: Dict[str, Any]) -> Optional[str]:
    """Why a Sales Agent report disagrees with the recorded sales, or None."""
    report = _check_keys(report, {"lines", "total_charged"}, "The report")
    sold = _current("sold_lines")
    for row in _check_rows(report["lines"], {"item_name", "quantity", "status", "line_total",
                                             "delivery_date", "reason"}, "lines"):
        key = (row["item_name"], _as_int(row["quantity"], "quantity"))
        line = sold.get(key)
        if row["status"] == "SOLD":
            if line is None:
                return f"{key[0]} x{key[1]} was not recorded by record_sale_tool, so it is not SOLD."
            if not _same_money(row["line_total"], line["line_total"]) or row["delivery_date"] != line["delivery"]:
                return (f"Report {key[0]} with the recorded figures: line_total {line['line_total']}, "
                        f"delivery_date {line['delivery']}.")
        elif row["status"] == "REJECTED":
            if line is not None:
                return f"{key[0]} x{key[1]} was recorded by record_sale_tool, so it is SOLD."
            if row["reason"] not in SALES_REASONS:
                return f"A REJECTED line needs a reason from {sorted(SALES_REASONS)}."
        else:
            return "status must be SOLD or REJECTED."
    reported = {(row["item_name"], int(row["quantity"])) for row in report["lines"] if row["status"] == "SOLD"}
    missing = [f"{item} x{quantity}" for item, quantity in sold if (item, quantity) not in reported]
    if missing:
        return f"These lines were recorded as sales and must be reported SOLD: {missing}."
    total = round(sum(line["line_total"] for line in sold.values()), 2)
    if not _same_money(report["total_charged"], total):
        return f"total_charged must be {total:.2f}, the sum of the SOLD lines."
    return None


def _report_check(source: str, find_problem):
    """Build a smolagents final-answer check from a report validator."""
    def check(final_answer: Any, memory: Any, agent: Any = None) -> bool:
        try:
            problem = find_problem(_parse_json_answer(final_answer))
        except ValueError as exc:
            problem = str(exc)
        if problem:
            _reject(source, problem)
        return True
    return check


inventory_report_check = _report_check("Inventory Agent", _inventory_report_problem)
quote_report_check = _report_check("Quoting Agent", _quote_report_problem)


def sales_report_check(final_answer: Any, memory: Any, agent: Any = None) -> bool:
    """Sales Agent final-answer check; also keeps each rejected line's reason for the reply."""
    _report_check("Sales Agent", _sales_report_problem)(final_answer, memory, agent)
    report = _parse_json_answer(final_answer)
    _current("rejected_lines").update({row["item_name"]: row["reason"]
                                       for row in report["lines"] if row["status"] == "REJECTED"})
    return True


# ---------------------------------------------------------------------------
# Customer reply: rendered in code from the Orchestrator's plan + the records
# ---------------------------------------------------------------------------

# Phrases that must never reach a customer: internal data or services we do not offer.
_INTERNAL_DETAIL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r"transaction[_ ]?ids?\b",
        r"\b\d[\d,]*\s*(?:units?|sheets?)?\s*(?:on hand|in stock|in inventory)\b",
        r"\bsuppliers?\b",
        r"\bvendors?\b",
        r"\bpartial shipments?\b",
        r"\bback-?orders?(?:ed)?\b",
        r"\b(?:markup|margin|cash balance|unit cost)\b",
        r"\b(?:inventory|quoting|sales|business_advisor)_agent\b",
        r"\b\w+_tool\b",
    )
]

# Customer-facing phrases that reveal stock counts or offer services the company
# did not perform (outside sourcing, expedited shipping, extra documents,
# substitutions). A sentence containing one is removed before the reply is sent.
CUSTOMER_BLOCKLIST = [
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r"\b\d[\d,]*\s+(?:units?|sheets?)\s+available\b",
        r"\b\d[\d,]*\s+(?:units?|sheets?)\s+(?:on hand|in stock|in inventory)\b",
        r"\b(?:source|sourcing|procure|procurement|special-order|special order|outside supplier|external(?:ly)?)\b",
        r"\b(?:expedited|shipping confirmation|tracking|invoice|PDF quote|packing slip|receipt)\b",
        r"\b(?:substitut\w*|alternatives?|equivalents?|lead[- ]time)\b",
    )
]

# Offers of follow-up work the system cannot carry out on a later turn (each
# request is handled once). They are replaced by CUSTOMER_CLOSING.
_CUSTOMER_OFFER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r"\bif you(?:'|’)?d like\b",
        r"\bif you (?:want|wish|can)\b",
        r"\bwould you like\b",
        r"\b(?:I|we) can (?!not\b|confirm\b)",
        r"\b(?:I|we)(?:'|’)ll\b",
        r"\blet (?:me|us) know\b",
        r"\btell (?:me|us) (?:which|how|if)\b",
        r"\bwhich (?:option )?(?:would )?you prefer\b",
        r"\breply with your choice\b",
    )
]

CUSTOMER_CLOSING = "To change anything, please send us a new request."

# Splits a reply into sentences, including after a closing quote or bracket,
# but not after the abbreviations "e.g." and "i.e.".
_SENTENCE_BREAK = re.compile(r"(?<!e\.g\.)(?<!i\.e\.)(?<=[.!?])\s+|(?<=[.!?][”\"’')])\s+")
_LIST_ITEM = re.compile(r"^\s*(?:[-•*]|\(?[a-z0-9]\)|\d+\.)\s")

# The Orchestrator's note is free text, so it must not carry figures or dates:
# every number the customer sees comes from the records.
_FIGURES_IN_NOTE = re.compile(r"\$\s?\d|\d{4}-\d{2}-\d{2}|\b\d[\d,]*\s+(?:units?|sheets?|reams?)\b", re.IGNORECASE)


def _customer_phrase_problems(reply: str) -> List[str]:
    """Blocked or offer phrases found in a customer reply."""
    return sorted({match.group(0) for pattern in CUSTOMER_BLOCKLIST + _CUSTOMER_OFFER_PATTERNS
                   for match in pattern.finditer(reply)})


def _is_blocked_sentence(sentence: str) -> bool:
    """True if the sentence must not reach the customer."""
    return any(pattern.search(sentence) for pattern in CUSTOMER_BLOCKLIST + _CUSTOMER_OFFER_PATTERNS)


def sanitize_customer_reply(reply: str) -> str:
    """Remove every sentence that leaks stock counts or offers unperformed services.

    Whole sentences are dropped (rather than words replaced) so the reply stays
    readable. When an offer such as "If you'd like, I can:" is dropped, the
    list of options under it goes too. If anything was removed, the reply ends
    with the standard closing inviting a new request.
    """
    kept_lines, removed, skipping_options = [], False, False
    for line in reply.splitlines():
        if skipping_options and _LIST_ITEM.match(line):
            continue
        skipping_options = False
        kept = []
        for sentence in _SENTENCE_BREAK.split(line):
            if _is_blocked_sentence(sentence):
                removed = True
                skipping_options = sentence.rstrip().endswith(":")
            else:
                kept.append(sentence)
        text = " ".join(kept).rstrip()
        # Drop a list bullet emptied by the filter, but keep blank spacing lines.
        if text.strip() not in {"", "-", "•", "*"} or not line.strip():
            kept_lines.append(text)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
    if removed and CUSTOMER_CLOSING not in cleaned:
        cleaned = f"{cleaned}\n\n{CUSTOMER_CLOSING}"
    return cleaned


def _plan_problem(plan: Dict[str, Any]) -> Optional[str]:
    """Why the Orchestrator's plan disagrees with the records or its note is unsafe, or None."""
    plan = _check_keys(plan, {"items", "note"}, "The final answer")
    note = str(plan["note"] or "")
    leaks = sorted({match.group(0) for pattern in _INTERNAL_DETAIL_PATTERNS for match in pattern.finditer(note)}
                   | set(_customer_phrase_problems(note)) | {m.group(0) for m in _FIGURES_IN_NOTE.finditer(note)})
    if leaks:
        return (f"The note contains {leaks}. Keep it to one or two friendly sentences with no prices, "
                "numbers, dates, internal details or offers of further services.")

    item_status, sold = _current("item_status"), _current("sold_lines")
    available = [name for name, status in item_status.items() if status == "AVAILABLE"]
    if available and "sales_agent" not in _current("agents_called"):
        return (f"Items {available} are AVAILABLE but the order was never finalized. "
                "Call quoting_agent (if not done) and then sales_agent first.")

    rows = _check_rows(plan["items"], {"requested", "quantity", "catalog_item", "outcome", "reason"}, "items")
    planned_sold = set()
    for row in rows:
        item, quantity = row["catalog_item"], _as_int(row["quantity"], "quantity")
        if item is not None and item not in item_status:
            return (f"'{item}' was never sent to inventory_agent. Send it, or set catalog_item to null "
                    "if it is not the same product.")
        if row["outcome"] == "SOLD":
            if (item, quantity) not in sold:
                return f"{item} x{quantity} was not sold by sales_agent: mark it NOT_SUPPLIED with a reason."
            planned_sold.add((item, quantity))
        elif row["outcome"] != "NOT_SUPPLIED" or row["reason"] not in PLAN_REASONS:
            return f"Each item is SOLD, or NOT_SUPPLIED with a reason from {sorted(PLAN_REASONS)}."
    missing_sold = [f"{item} x{quantity}" for item, quantity in sold if (item, quantity) not in planned_sold]
    if missing_sold:
        return f"These lines were sold and must be listed as SOLD: {missing_sold}."
    missing_items = [name for name in item_status if name not in {row["catalog_item"] for row in rows}]
    if missing_items:
        return f"Every item sent to the team must be listed; missing: {missing_items}."
    return None


plan_matches_records = _report_check("Orchestrator", _plan_problem)


def _long_date(iso_date: str) -> str:
    """'2025-04-12' -> 'April 12, 2025'."""
    day = datetime.strptime(iso_date, "%Y-%m-%d")
    return f"{day:%B} {day.day}, {day.year}"


def _unit_price_text(unit_price: float) -> str:
    """List prices have up to 3 decimals: 0.25 -> '$0.25', 0.188 -> '$0.188'."""
    text_price = f"{unit_price:.3f}"
    return f"${text_price[:-1] if text_price.endswith('0') else text_price}"


def _unsupplied_reason_code(catalog_item: Optional[str], plan_reason: Optional[str]) -> str:
    """Why an item was not supplied, from the records where they say why."""
    status = _current("item_status").get(catalog_item)
    if catalog_item is None:
        return "not_in_catalog"
    if status == "NOT_STOCKED":
        return "not_stocked"
    if status in ("RESTOCK_NEEDED", "UNAVAILABLE"):
        return "cannot_supply_in_time"
    code = _current("rejected_lines").get(catalog_item) or plan_reason
    return code if code in PLAN_REASONS else "not_available"


def _unsupplied_reason(catalog_item: Optional[str], plan_reason: Optional[str]) -> str:
    """Customer-facing sentence for why an item was not supplied."""
    deliver_by = _long_date(_current("deliver_by"))
    return {
        "not_in_catalog": "this is not in our catalog",
        "not_stocked": "we do not stock this item",
        "cannot_supply_in_time": f"we cannot have enough in stock by your date of {deliver_by}",
        "cannot_deliver_by_deadline": f"it cannot be delivered by your date of {deliver_by}",
    }.get(_unsupplied_reason_code(catalog_item, plan_reason), "it is not available for this order")


def _requested_label(requested: str, quantity: int) -> str:
    """The customer's wording plus the quantity, unless the wording already states it."""
    if re.search(rf"\b(?:{quantity}|{quantity:,})\b", requested):
        return requested
    return f"{requested} ({quantity:,} requested)"


def render_customer_reply(plan: Optional[Dict[str, Any]]) -> str:
    """Write the customer reply from the recorded sales and the Orchestrator's plan.

    Prices, the discount rationale, the total and delivery dates always come
    from the request records, so the reply can never state a figure that
    differs from the database. The plan adds the customer's own wording, the
    items we could not supply and a short personal note. With no valid plan
    (e.g. the Orchestrator ran out of steps), the reply is built from the
    records alone.
    """
    sold, quotes = _current("sold_lines"), _current("quote_lines")
    rows = plan["items"] if plan else []
    requested_as = {(row["catalog_item"], int(row["quantity"])): row["requested"] for row in rows}

    note = str(plan["note"]).strip() if plan else ""
    # The note often thanks the customer already; avoid "Thank you ... Thank you".
    opening = note if note.lower().startswith("thank") else f"Thank you for your request. {note}".strip()
    lines = [opening]

    if sold:
        lines += ["", "Your confirmed order:"]
        rate = _current("discount_rate")
        for (item, quantity), line in sold.items():
            asked = requested_as.get((item, quantity), item)
            asked_text = f' (you asked for "{asked}")' if asked.strip().lower() != item.lower() else ""
            quote = quotes.get(item)
            price_text = f" at {_unit_price_text(quote['unit_price'])} each" if quote else ""
            discount_text = f", less {rate:.0%} bulk discount" if quote and rate > 0 else ""
            lines.append(f"- {item}{asked_text}: {quantity:,} units{price_text}{discount_text} = "
                         f"${line['line_total']:,.2f}, delivered by {_long_date(line['delivery'])}")
        if _current("quote_rationale"):
            lines.append(f"Discount: {_current('quote_rationale')}")
        lines.append(f"Total charged: ${sum(line['line_total'] for line in sold.values()):,.2f}.")

    if plan:
        unsupplied = [f"- {_requested_label(str(row['requested']), _as_int(row['quantity'], 'quantity'))}: "
                      f"{_unsupplied_reason(row['catalog_item'], row['reason'])}."
                      for row in rows if row["outcome"] != "SOLD"]
    else:
        sold_items = {item for item, _ in sold}
        unsupplied = [f"- {item}: {_unsupplied_reason(item, None)}."
                      for item in _current("item_status") if item not in sold_items]
        unsupplied.append("- Any other items in your request: these are not in our catalog.")
    if unsupplied:
        lines += ["", "We could not supply:"] + unsupplied

    if not sold:
        lines += ["", "We are sorry we could not help this time. Nothing has been charged."]
    lines += ["", CUSTOMER_CLOSING]
    return "\n".join(lines)


# max_tool_threads=1 keeps tool calls sequential so database writes never race.
inventory_agent = WorkerAgent(
    tools=[check_stock_tool, inventory_snapshot_tool, check_cash_tool,
           supplier_lead_time_tool, reorder_stock_tool],
    model=model,
    name="inventory_agent",
    description=("Checks stock for exact catalog items and reorders from the supplier when needed. "
                 "Give it lines 'exact item name | quantity'. Returns JSON: "
                 '{"items": [{item_name, quantity, status AVAILABLE/UNAVAILABLE, reason}]}.'),
    instructions=INVENTORY_INSTRUCTIONS,
    final_answer_checks=[inventory_report_check],
    display_name="Inventory Agent",
    max_steps=15,
    max_tool_threads=1,
    verbosity_level=AGENT_VERBOSITY,
)

quoting_agent = WorkerAgent(
    tools=[quote_history_tool, price_lookup_tool, calculate_quote_tool],
    model=model,
    name="quoting_agent",
    description=("Prices available items with bulk discounts using quote history. "
                 "Give it lines 'exact item name | quantity' and the event type if known. Returns JSON: "
                 '{"lines": [{item_name, quantity, unit_price, line_total}], discount_rate, quote_total, rationale}.'),
    instructions=QUOTING_INSTRUCTIONS,
    final_answer_checks=[quote_report_check],
    display_name="Quoting Agent",
    max_steps=8,
    max_tool_threads=1,
    verbosity_level=AGENT_VERBOSITY,
)

sales_agent = WorkerAgent(
    tools=[delivery_check_tool, record_sale_tool],
    model=model,
    name="sales_agent",
    description=("Finalizes an order: checks each line can be delivered by the deadline and records the sale. "
                 "Give it lines 'exact item name | quantity | line_total'. Returns JSON: "
                 '{"lines": [{item_name, quantity, status SOLD/REJECTED, line_total, delivery_date, reason}], '
                 "total_charged}."),
    instructions=SALES_INSTRUCTIONS,
    final_answer_checks=[sales_report_check],
    display_name="Sales Agent",
    max_steps=12,
    max_tool_threads=1,
    verbosity_level=AGENT_VERBOSITY,
)

business_advisor_agent = ToolCallingAgent(
    tools=[financial_report_tool, cash_trend_tool, stock_health_tool],
    model=model,
    name="business_advisor_agent",
    description="Read-only analyst that reviews finances and stock and recommends operational changes.",
    instructions=ADVISOR_INSTRUCTIONS,
    max_steps=6,
    max_tool_threads=1,
    verbosity_level=AGENT_VERBOSITY,
)

orchestrator_agent = ToolCallingAgent(
    tools=[match_catalog_items_tool],
    managed_agents=[inventory_agent, quoting_agent, sales_agent],
    model=model,
    name="orchestrator_agent",
    description="Handles customer requests by delegating to the inventory, quoting and sales agents.",
    instructions=ORCHESTRATOR_INSTRUCTIONS,
    final_answer_checks=[plan_matches_records],
    max_steps=12,
    max_tool_threads=1,
    verbosity_level=AGENT_VERBOSITY,
)


# ---------------------------------------------------------------------------
# Entry points used by run_test_scenarios()
# ---------------------------------------------------------------------------

CUSTOMER_FALLBACK_REPLY = (
    "Thank you for your request. We were unable to complete it right now and "
    "nothing has been charged. Please contact our sales team and we will be glad to help."
)


# Every item a customer asked for that was not sold, across the run: structured
# evidence for the Business Advisor's range and stock recommendations.
_lost_demand: List[Dict[str, Any]] = []


def reset_business_advisor_report() -> None:
    """Start a fresh advisor report and lost-demand log for a new test run."""
    _lost_demand.clear()
    # utf-8-sig writes a BOM once at the start so Windows tools detect UTF-8.
    with open(ADVISOR_REPORT_PATH, "w", encoding="utf-8-sig") as report_file:
        report_file.write(f"Business Advisor report - run started {datetime.now():%Y-%m-%d %H:%M}\n\n")


def _record_lost_demand(plan: Optional[Dict[str, Any]], request_date: str) -> List[Dict[str, Any]]:
    """Log this request's unsold items (with reason codes) and return them."""
    if plan:
        unsold = [(str(row["requested"]), row["catalog_item"], int(row["quantity"]), row["reason"])
                  for row in plan["items"] if row["outcome"] != "SOLD"]
    else:
        sold_items = {item for item, _ in _current("sold_lines")}
        unsold = [(item, item, None, None) for item in _current("item_status") if item not in sold_items]
    lost = [{"request_date": request_date, "requested": requested, "catalog_item": catalog_item,
             "units": quantity, "reason": _unsupplied_reason_code(catalog_item, reason)}
            for requested, catalog_item, quantity, reason in unsold]
    _lost_demand.extend(lost)
    return lost


def lost_demand_summary() -> List[Dict[str, Any]]:
    """Lost demand grouped by item and reason, most units first."""
    totals: Dict[tuple, Dict[str, Any]] = {}
    for line in _lost_demand:
        item = line["catalog_item"] or line["requested"]
        entry = totals.setdefault((item, line["reason"]),
                                  {"item": item, "reason": line["reason"], "requests": 0, "units": 0})
        entry["requests"] += 1
        entry["units"] += line["units"] or 0
    return sorted(totals.values(), key=lambda entry: (-entry["units"], entry["item"]))


def _run_business_review(context: str, review_date: str, heading: str) -> None:
    """Hand an outcome to the Business Advisor and log its internal notes."""
    progress.event("Orchestrator", "sharing outcome with Business Advisor", "handoff")
    try:
        notes = str(business_advisor_agent.run(f"Review date: {review_date}\n\n{context}"))
        progress.event("Business Advisor", "recommendations logged", "ok")
    except Exception as exc:
        notes = f"(review failed: {exc})"
        progress.event("Business Advisor", "review failed", "fail")
    with open(ADVISOR_REPORT_PATH, "a", encoding="utf-8") as report_file:
        report_file.write(f"=== {heading} ===\n{notes}\n\n")


def _validated_plan(answer: Any) -> Optional[Dict[str, Any]]:
    """The Orchestrator's plan if it is valid JSON that matches the records, else None.

    Re-checked here because an agent that runs out of steps returns an
    answer that never went through its final-answer check.
    """
    try:
        plan = _parse_json_answer(answer)
        return None if _plan_problem(plan) else plan
    except ValueError:
        return None


def call_your_multi_agent_system(request_text: str) -> str:
    """Handle one customer request end to end and return the customer reply."""
    request_date = _extract_request_date(request_text)
    deliver_by = _extract_deliver_by(request_text, request_date)
    start_request_context(request_date, deliver_by)
    task = (f"Request date: {request_date}. Deliver-by date: {deliver_by}.\n\n"
            f"Customer request:\n{request_text}")
    lost: List[Dict[str, Any]] = []
    with progress.track(f"Processing request dated {request_date}..."):
        progress.event("Customer", f"request received (deliver by {deliver_by})", "handoff")
        try:
            plan = _validated_plan(orchestrator_agent.run(task))
            if plan is None:
                progress.event("Orchestrator", "no valid plan - reply built from the records alone", "fail")
            # Final safety net: no stock counts or unperformed services reach the customer.
            reply = sanitize_customer_reply(render_customer_reply(plan))
            lost = _record_lost_demand(plan, request_date)
            progress.event("Orchestrator", "reply sent to customer", "ok")
        except Exception as exc:
            # Never show internal errors to the customer.
            progress.event("Orchestrator", f"failed: {exc}", "fail")
            reply = CUSTOMER_FALLBACK_REPLY
        if ADVISOR_REVIEW_EVERY_REQUEST:
            _run_business_review(
                f"Customer request:\n{request_text}\n\nReply sent to the customer:\n{reply}\n\n"
                f"Lost demand in this request (structured):\n{_to_json(lost)}",
                request_date,
                f"Request dated {request_date}",
            )
    return reply


def run_final_business_review(start_date: str, end_date: str) -> None:
    """End-of-run review of all requests by the Business Advisor, with the lost-demand log."""
    summary = json.dumps(lost_demand_summary(), indent=2)
    with open(ADVISOR_REPORT_PATH, "a", encoding="utf-8") as report_file:
        report_file.write(f"=== Lost demand log {start_date} to {end_date} (by item and reason) ===\n{summary}\n\n")
    with progress.track("Business Advisor: end-of-run review..."):
        _run_business_review(
            f"End-of-run review covering {start_date} to {end_date}. Use before_date = {start_date} "
            f"and after_date = {end_date} for cash_trend_tool.\n\n"
            f"Lost demand across all requests (structured, by item and reason):\n{summary}",
            end_date,
            f"End-of-run review {start_date} to {end_date}",
        )


# Run your test scenarios by writing them here. Make sure to keep track of them.


def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    # Agents are built at import time; only the advisor log needs a fresh start.
    reset_business_advisor_report()

    results = []
    for request_number, (_, row) in enumerate(quote_requests_sample.iterrows(), start=1):
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {request_number} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        multi_agent_system = globals().get("call_your_multi_agent_system")
        response = (
            multi_agent_system(request_with_date)
            if callable(multi_agent_system)
            else "Multi-agent system is not configured."
        )

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": request_number + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Business Advisor's end-of-run recommendations (internal only).
    run_final_business_review(initial_date, final_date)
    print(f"Business Advisor recommendations saved to {ADVISOR_REPORT_PATH}")

    # Save results
    # utf-8-sig adds a BOM so Excel reads replies' dashes/quotes as UTF-8, not cp1252.
    pd.DataFrame(results).to_csv("test_results.csv", index=False, encoding="utf-8-sig")
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
