# Starter Code Review

Step 2 review of `project_starter.py`: what each provided function does, how
the system will use it, and the details that shaped the tool design in
[[Agent Workflow]].

## Setup already in the file

- **Framework:** chosen in Step 3 (see below). I added the `smolagents`
  imports (`ToolCallingAgent`, `OpenAIServerModel`, `tool`), the model setup
  and the `smolagents==1.26.0` pin in `requirements.txt` myself; the starter
  code did not preselect a framework.
- **Model:** `gpt-5-mini` (started on `gpt-4o-mini`) through the Vocareum
  OpenAI proxy but there were many errors pointing to 4o being underpowered,
   with the key read from `UDACITY_OPENAI_API_KEY` in `.env`.
- **Database:** SQLite file `munder_difflin.db` through a SQLAlchemy engine.
- **`paper_supplies`:** the full catalog of 46 items, each with `item_name`,
  `category` (paper / product / large_format / specialty) and `unit_price`.
  This is the only place prices exist for items that are not stocked.

## Framework choice (Step 3): smolagents

### Why smolagents fits this project

1. **Least setup for this project.** Wiring it in took one import line, one
   pinned dependency and a five-line model object. The rest of the starter
   code (pandas, SQLAlchemy, plain Python helpers) needed no changes.
2. **The proxy works with no extra adapter.** `OpenAIServerModel(api_base=...)`
   points straight at the Vocareum OpenAI-compatible endpoint with the key
   from `.env`.
3. **Orchestrator-worker is built in.** Any agent given a `name` and
   `description` can be passed to another agent through `managed_agents=[...]`
   (confirmed in the installed 1.26.0). The Orchestrator then calls the
   Inventory, Quoting, Sales and Advisor agents like tools, which is exactly
   the shape of [[Agent Workflow]].
4. **Helpers become tools with little code.** `@tool` builds the tool schema
   from type hints and the docstring's `Args:` section. The starter helpers
   already use that docstring style, so each tool wrapper is a few lines.
5. **`ToolCallingAgent` is the safer agent type here.** It makes structured
   JSON tool calls. smolagents' other type, `CodeAgent`, writes and runs
   Python, which is riskier next to a database that `create_transaction`
   writes to.
6. **Easy to debug.** Each run logs every step (the model's reasoning, tool
   calls, arguments and results), which helps find why an order was rejected
   or priced wrongly.
7. **Single-file submission.** Agents, tools and wiring all live in Python
   code, so everything fits in the one `.py` file the project asks for.

### Why not pydantic-ai

- Its main strength is **typed, validated outputs** (Pydantic models), which
  would suit quotes. It is a reasonable second choice.
- Pointing it at the Vocareum proxy needs a separate provider object as well
  as the model, and every tool and output type adds Pydantic schema code.
- Multi-agent work is done by **calling one agent from inside another
  agent's tool** (agent delegation). That works, but the orchestrator-worker
  pattern takes more hand-written glue than smolagents' `managed_agents`.

### Why not npcsh

- npcsh is built around an **agentic command-line shell** and NPC "teams"
  usually defined in YAML files on disk. That design suits interactive shell
  use more than a batch script processing a CSV of requests.
- Agents defined in separate files clash with the **one Python file**
  submission rule.
- It has a smaller community and fewer examples than the other two, so
  debugging help is harder to find on a 6-hour project.

### Trade-off to watch

smolagents does not validate the final answer's structure. To keep the
customer reply consistent, the Quoting tool (`calculate_quote_tool`) returns
the numbers and rationale as plain Python, and the Orchestrator's prompt
fixes the reply format.

## Function descriptions

### Database setup (used by the test harness, not by agents)

| Function | What it does | How the system uses it |
| :--- | :--- | :--- |
| `generate_sample_inventory(paper_supplies, coverage=0.4, seed=137)` | Picks a random 40% of the catalog (seeded, so always the same items) and gives each a starting stock of 200–799 and a `min_stock_level` of 50–149. Returns a DataFrame. | Called inside `init_database`. Defines which items are "stocked" and their reorder thresholds. |
| `init_database(db_engine, seed=137)` | Builds the four tables: `transactions` (empty schema), `quote_requests` (from `quote_requests.csv`), `quotes` (from `quotes.csv`, with `job_type` / `order_size` / `event_type` unpacked) and `inventory`. Seeds a $50,000 starting cash balance as a dummy `sales` row and one `stock_orders` row per stocked item, all dated 2025-01-01. | Called once at the start of `run_test_scenarios()` to reset the DB. |

### Helper functions (all 7 must back an agent tool)

| Function | What it does | Returns | Tool / agent |
| :--- | :--- | :--- | :--- |
| `create_transaction(item_name, transaction_type, quantity, price, date)` | Inserts one row into `transactions`. `transaction_type` must be `'stock_orders'` (buying from the supplier, cash goes down) or `'sales'` (selling to a customer, cash goes up). **`price` is the total for the line, not the unit price.** | New row id (`int`) | `reorder_stock_tool` (Inventory), `record_sale_tool` (Sales) |
| `get_all_inventory(as_of_date)` | Net stock per item (stock orders minus sales) up to and including the date. **Only items with stock > 0 are returned.** | `{item_name: stock}` | `inventory_snapshot_tool` (Inventory), `stock_health_tool` (Advisor) |
| `get_stock_level(item_name, as_of_date)` | Net stock for one item up to the date. Returns 0 for an unknown or never-stocked item, rather than an error. | 1-row DataFrame: `item_name`, `current_stock` | `check_stock_tool` (Inventory); stock guard inside `record_sale_tool` (Sales) |
| `get_supplier_delivery_date(input_date_str, quantity)` | Supplier lead time by quantity: ≤10 same day, 11–100 → +1 day, 101–1000 → +4 days, >1000 → +7 days. Falls back to today on a bad date. Prints a debug line on every call. | `'YYYY-MM-DD'` | `supplier_lead_time_tool` (Inventory), `delivery_check_tool` (Sales) |
| `get_cash_balance(as_of_date)` | Sum of all `sales` prices minus all `stock_orders` prices up to the date. Includes the $50,000 seed. Returns 0.0 on error. | `float` | `check_cash_tool` (Inventory), `cash_trend_tool` (Advisor) |
| `generate_financial_report(as_of_date)` | Cash balance, inventory value, total assets, per-item stock and value, and the top 5 sellers by revenue. **Values only items in the `inventory` table.** | `dict` | `financial_report_tool` (Advisor); also called by the harness after every request |
| `search_quote_history(search_terms, limit=5)` | `LIKE` search over past customer requests and quote explanations, newest first. **Terms are combined with AND** (the docstring says "any", but the SQL uses `AND`), so more terms means fewer hits. | List of dicts: `original_request`, `total_amount`, `quote_explanation`, `job_type`, `order_size`, `event_type`, `order_date` | `quote_history_tool` (Quoting) |

### Test harness

| Function | What it does |
| :--- | :--- |
| `run_test_scenarios()` | Resets the DB, loads `quote_requests_sample.csv` (20 requests, 2025-04-01 to 2025-04-17), sorts by date and sends each one as `"<request> (Date of request: YYYY-MM-DD)"` to a global function named **`call_your_multi_agent_system`**. After each request it reruns `generate_financial_report`, prints cash and inventory, and writes everything to `test_results.csv`. |

## Gotchas that shape the design

1. **Exact item names are required.** Every DB query matches on `item_name`
   exactly. Customers write "A4 sheets" or "heavy cardstock", so requests must
   be mapped to catalog names first or stock will read as 0.
2. **Only 40% of the catalog is stocked.** Prices for the rest exist only in
   `paper_supplies`, so the price lookup reads that list, not the DB.
3. **Unstocked items are invisible to the financial report.**
   `generate_financial_report` reads the `inventory` table, so buying an
   unstocked item lowers cash but adds no inventory value. Decision: reorder
   only items in the `inventory` table, and decline unstocked items with a
   clear reason.
4. **Reorder level comes from the `inventory` table.** `min_stock_level` is
   not returned by any helper, so `check_stock_tool` reads it with one small
   query next to `get_stock_level`.
5. **`create_transaction` takes the total price.** Tools compute
   `quantity × unit_price` (minus any discount for sales) before calling it.
6. **Dates are compared as strings.** Every tool passes plain `YYYY-MM-DD`
   dates. A reorder is dated on the request date so the new stock counts
   before the sale on the same date; the supplier lead time is then checked
   against the customer's deadline.
7. **Quote history search uses AND.** `quote_history_tool` searches with one
   or two keywords (e.g. `["cardstock"]`, `["ceremony"]`), not the full item
   list.
8. **Historic discount practice.** `quotes.csv` explanations show bulk
   discounts around 5–15% and rounded "friendly" totals. The Quoting Agent
   copies this pattern.
9. **The harness's `request_id` is `request_number + 1`**, so IDs in
   `test_results.csv` start at 2. This is cosmetic; mention it in the report
   instead of changing the harness.
10. **Delivery timing.** The only timing helper is the supplier lead time.
    The Sales Agent uses `get_supplier_delivery_date(request_date, quantity)`
    as the delivery estimate and compares it with the customer's deliver-by
    date.
