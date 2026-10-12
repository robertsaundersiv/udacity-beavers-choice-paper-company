# Reflection Report: Beaver's Choice Multi-Agent System

Multi-agent quoting, inventory and order-fulfillment system for the Beaver's
Choice Paper Company, built with **smolagents** and evaluated on the 20
requests in `quote_requests_sample.csv`.

## Results at a glance

| Rubric requirement | Result |
| :--- | :--- |
| At least 3 requests change the cash balance | **13 of 20** (12 up, 1 down) |
| At least 3 quote requests successfully fulfilled | **12 of 20** sold at least one line; 3 fully fulfilled |
| Not every request fulfilled, with reasons given | **8 of 20** sold nothing; every unsold line has a stated reason |
| Every helper function used in a tool | All 7 (table in section 1.4) |
| Maximum of five agents | 5: Orchestrator + 4 workers |

Over the run, revenue was **$2,126.62** from 19 order lines (16,650 units).
Total assets rose from **$50,000.00 to $50,319.12**. That increase equals the
gross profit exactly: no sale was made below cost.

---

## 1. The multi-agent system

### 1.1 Architecture

![Agent workflow diagram](Agent%20Workflow.svg)

Files: `Agent Workflow.svg` (architecture), `Agent Workflow Sequence.svg`
(step-by-step flow for one request) and `Agent Workflow.md` (both diagrams as
Mermaid, plus the full tool reference).

| Agent | Responsibility | Tools |
| :--- | :--- | :--- |
| **Orchestrator** | Reads the request, maps items to exact catalog names, delegates to Inventory, Quoting and Sales in order, writes the customer reply | `match_catalog_items_tool` |
| **Inventory Agent** | Decides whether each item can be supplied; restocks from the supplier when cash and lead time allow | `check_stock_tool`, `inventory_snapshot_tool`, `check_cash_tool`, `supplier_lead_time_tool`, `reorder_stock_tool` |
| **Quoting Agent** | Prices available items with list prices and bulk discounts, guided by quote history | `quote_history_tool`, `price_lookup_tool`, `calculate_quote_tool` |
| **Sales Agent** | Checks each line can be delivered by the customer's date and records the sale | `delivery_check_tool`, `record_sale_tool` |
| **Business Advisor** | Read-only analyst: reviews cash, stock and the outcome after every reply and at the end of the run | `financial_report_tool`, `cash_trend_tool`, `stock_health_tool` |

Responsibilities do not overlap. Only Inventory buys stock, only Sales records
sales, only Quoting sets prices, and the Advisor writes nothing to the
database. The Orchestrator has no database tools at all.

### 1.2 How a request flows

1. The orchestration layer (`call_your_multi_agent_system`) parses the
   **request date** and the **deliver-by date** from the text in code. It
   stores them in a per-request context that every database tool reads.
2. The **Orchestrator** lists the items and calls `match_catalog_items_tool`.
   This is a TF-IDF character n-gram match over the 46-item catalog. The
   Orchestrator picks the candidate that is genuinely the same product, or
   marks the item "not in our catalog" (e.g. A3 paper, balloons, tickets).
3. **Inventory** checks each item. It returns NOT_STOCKED if the company does
   not carry it, and AVAILABLE if stock covers the order. If stock is short,
   it checks supplier lead time and cash, and restocks only if the delivery
   arrives by the customer's date.
4. **Quoting** prices the available items. List price is supplier cost plus
   a 25% markup. Bulk discounts are 5% from 1,000 units, 10% from 5,000 and
   15% from 10,000, with a price floor of cost plus 5%.
5. **Sales** re-checks the deadline and stock for each line, then records it
   with `create_transaction('sales')`.
6. The **Orchestrator** replies to the customer: what was sold, prices, the
   discount and why, the total charged, the delivery date, and a reason for
   every item that could not be supplied.
7. The **Business Advisor** reviews the outcome and appends recommendations
   to `business_advisor_report.txt`. These notes are internal and are never
   sent to the customer.

### 1.3 Design decisions and why

**Five agents, split by business function.** The split mirrors how a small
distributor works: customer service, stock control, pricing and order
processing are separate jobs. Keeping them apart means each agent has a short
prompt and only the tools it needs. That matters for a small model: fewer
tools per agent meant fewer wrong tool choices. The fifth slot went to the
Business Advisor (a "stand out" suggestion) rather than a negotiating
customer agent, which would have been a sixth agent.

**The Advisor runs after the reply, not as an Orchestrator tool.** If the
Orchestrator could call the Advisor, its internal notes could leak into the
customer reply. Running it afterwards keeps those notes out of every reply.

**Rules that matter are enforced in code, not prompts.** Early test runs
(first on `gpt-4o-mini`) showed failures that prompting alone did not fix. Each
became a code-level guardrail:

| Failure seen in testing | Guardrail |
| :--- | :--- |
| Agents dropped or invented dates (one used 2023-10-03), so stock and cash read as $0 | Dates parsed in code; database tools take no date arguments |
| Customer wording ("A4 glossy paper") sent to Inventory instead of catalog names | Tools reject non-catalog names and list the closest exact names |
| A sale recorded after its own delivery check said it would be late | `record_sale_tool` re-checks the deadline itself |
| Orchestrator skipped Sales, then told the customer they were charged $40.50 | smolagents `final_answer_checks`: available items must reach Sales |
| Reply total or delivery dates differed from the database | Same check: reply must state the recorded total and dates, or say nothing was charged |
| Replies leaked stock counts, supplier details and transaction ids | Same check rejects internal-detail phrases; the agent rewrites |
| Repeated delegation could double-book a sale | Sales and customer restocks are idempotent within a request |
| Parallel tool calls could race on SQLite | `max_tool_threads=1` on every agent |

**Pricing with a markup.** `init_database` values stock and records purchases
at catalog `unit_price`. Selling at that same price with a bulk discount
would lose money on every restocked sale. A 25% markup with discounts capped
at 15% keeps every sale above cost. The markup is never shown to customers.

**Framework and model.** smolagents was chosen for its built-in managed-agent
delegation (`managed_agents`), `@tool` schemas generated from docstrings, and
`final_answer_checks`. It also connects to the course's OpenAI-compatible
proxy without an adapter. The final model is `gpt-5-mini` with
`reasoning_effort='low'`: it followed the multi-step procedure far more
reliably than `gpt-4o-mini`, and low effort cut each request to roughly
25–90 seconds.

**Terminal animation (stand-out feature).** Every tool call and hand-off
emits a status event to a live spinner and timeline (`ProgressDisplay`, built
on `rich`), so you can watch each request move through the agents.

### 1.4 Helper function coverage

| Starter helper | Used in tool(s) |
| :--- | :--- |
| `create_transaction` | `reorder_stock_tool` (`stock_orders`), `record_sale_tool` (`sales`) |
| `get_all_inventory` | `inventory_snapshot_tool`, `stock_health_tool` |
| `get_stock_level` | `check_stock_tool`, `record_sale_tool` |
| `get_supplier_delivery_date` | `supplier_lead_time_tool`, `reorder_stock_tool`, `delivery_check_tool`, `record_sale_tool` |
| `get_cash_balance` | `check_cash_tool`, `reorder_stock_tool`, `cash_trend_tool` |
| `generate_financial_report` | `financial_report_tool` |
| `search_quote_history` | `quote_history_tool` |

---

## 2. Evaluation results (`test_results.csv`)

The run processed all 20 sample requests (2025-04-01 to 2025-04-17). The
`request_id` column starts at 2 because of an off-by-one in the provided
harness (`request_number + 1`); it was left unchanged.

### 2.1 Per-request outcomes

"Lines sold" counts order lines sold out of lines requested. "Cash change" is
revenue minus any same-day restock purchases, which is why it is often lower
than the amount charged.

| ID | Date | Lines sold | Charged | Cash change | Unsold lines and reason |
| :--- | :--- | :--- | ---: | ---: | :--- |
| 2 | 04-01 | 3 / 3 | $81.30 | +$81.30 | none |
| 3 | 04-03 | 0 / 3 | $0.00 | $0.00 | poster paper, streamers (not stocked); balloons (not in catalog) |
| 4 | 04-04 | 1 / 3 | $527.00 | +$33.85 | A3 paper, "printer paper" (not matched); A4 restocked 9,863 and sold with 15% discount |
| 5 | 04-05 | 2 / 2 | $109.50 | +$74.05 | none |
| 6 | 04-05 | 2 / 3 | $118.90 | +$73.90 | washi tape (not stocked) |
| 7 | 04-06 | 1 / 3 | $37.60 | +$7.60 | construction paper, printer paper (not stocked) |
| 8 | 04-07 | 3 / 4 | $510.72 | +$428.72 | A3 matte (not in catalog); 5% discount at 1,000 units |
| 9 | 04-07 | 1 / 4 | $125.00 | +$25.00 | matte, A5, recycled paper (not stocked / not in catalog) |
| 10 | 04-07 | 0 / 3 | $0.00 | $0.00 | A4 cannot arrive by 04-10; A3 glossy, envelopes not carried |
| 11 | 04-08 | 0 / 2 | $0.00 | $0.00 | both lines cannot arrive by 04-10 |
| 12 | 04-08 | 1 / 3 | $37.60 | +$7.60 | copy paper, napkins (not stocked) |
| 13 | 04-08 | 2 / 2 | $181.40 | +$36.40 | none |
| 14 | 04-08 | 1 / 2 | $125.00 | +$25.00 | matte paper (not stocked) |
| 15 | 04-09 | 1 / 3 | $94.00 | +$19.00 | A4 x5,000 late (04-16); poster paper (not stocked) |
| 16 | 04-12 | 0 / 3 | $0.00 | $0.00 | A4 x10,000 and cardstock late; A3 not in catalog |
| 17 | 04-13 | 0 / 3 | $0.00 | $0.00 | A4 late; construction paper, poster board not carried |
| 18 | 04-14 | 0 / 5 | $0.00 | −$50.00 | A4 and plates late; A3, napkins, cups not carried |
| 19 | 04-14 | 0 / 3 | $0.00 | $0.00 | cardstock, colored paper late; copy paper not stocked |
| 20 | 04-15 | 1 / 3 | $178.60 | +$28.60 | glossy x2,000 late (needs 7 days); A3 matte not in catalog |
| 21 | 04-17 | 0 / 3 | $0.00 | $0.00 | flyers, posters (not stocked); tickets (not in catalog) |

### 2.2 Line-level summary

| Outcome | Lines | Share |
| :--- | ---: | ---: |
| Sold | 19 | 32% |
| Rejected: could not be delivered by the customer's date | 12 | 20% |
| Rejected: item not stocked or not in the catalog | 29 | 48% |
| **Total requested** | **60** | |

| Finance | Amount |
| :--- | ---: |
| Revenue | $2,126.62 |
| Cost of goods sold (at supplier cost) | $1,807.50 |
| Gross profit (15.0% margin) | $319.12 |
| Restock purchases | $1,335.60 (15 orders) |
| Cash | $45,059.70 → $45,850.72 |
| Inventory value | $4,940.30 → $4,468.40 |

### 2.3 Strengths

- **Replies match the books.** The totals in customer replies add up to
  $2,126.62, exactly the revenue in the `transactions` table. No reply claims
  a charge that was not recorded, and every no-sale reply says nothing was
  charged. This is enforced by the reply check, not left to the model.
- **Impossible requests are rejected with clear reasons.** Requests 10 and 11
  ask for 101–1,000 units within 2–3 days, but the supplier needs 4. Both
  were declined with the earliest possible date, and nothing was charged.
- **Correct bulk pricing.** Discounts were applied exactly at their tiers:
  15% on 10,000 A4 sheets (request 4) and 5% at exactly 1,000 units
  (requests 8 and 20). The arithmetic checks line by line. Replies explain
  why a discount did or did not apply ("discounts begin at 1,000 units").
- **Profitable and solvent.** Every sale cleared the price floor, giving a
  15.0% gross margin. Restocks were checked against cash before buying.
- **Stock kept above minimum.** Inventory tops items back up to
  `min_stock_level` after sales. The Advisor's end-of-run review found no
  item below its minimum or out of stock.
- **Large restocks handled end to end.** Request 4 needed 10,000 A4 sheets
  against 272 in stock. Inventory confirmed the 7-day lead time met the
  deadline, bought 9,863 sheets, and the sale cleared at a profit.

### 2.4 Areas for improvement

- **Too few items stocked (48% of lost lines).** Only 18 of 46 catalog items
  are stocked (fixed by the starter's seeded inventory). Demand clustered on
  items the company does not carry: standard copy paper, poster paper,
  matte paper, construction paper and napkins each appeared more than once.
- **Lean stock turns into missed deadlines (20% of lost lines).** Restocks
  only bring stock back to the minimum level. Cardstock and glossy paper
  therefore sat at about 150 sheets, and almost every mid-size order needed a
  same-day restock (14 of the 15 restocks were bought the same day as a
  sale of that item), which carries a 4–7 day lead time. From 04-12
  onward, against 04-15 deadlines, those lead times rejected every order
  (requests 16–19). In addition, `delivery_check_tool` uses the
  supplier's lead time even for lines already in stock, which overstates
  delivery times.
- **A guardrail gap in restocking.** In request 18, Inventory bought 1,000 A4
  sheets as a "routine top-up". That type of restock is not checked against
  the deadline, so it bypassed the check that refuses late customer
  restocks. The line was then rejected for its deadline anyway, so the $50
  became surplus stock. It wasn't lost, but it wasn't needed.
- **Reply wording.** Some replies still offer services the company does not
  provide ("special order", "source externally", "schedule a later date").
  Some mislabel reasons: request 16 calls A3 paper late rather than not in
  the catalog, and request 21 says Flyers is not in the catalog when it is
  in the catalog but not stocked. One reply (request 21) revealed a stock
  count ("399 units available") using a phrase the leak check does not
  match.
- **Inconsistent item matching.** "Printer paper" was sometimes matched to A4
  paper (request 5) and sometimes to standard copy paper or no match
  (requests 4, 7, 12). Catalog matching is left to the model's judgment and
  varies between requests.
- **Generic advisor output.** The Advisor's recommendations are grounded in
  real numbers but miss the biggest issues above (range gaps and lead
  times). It also measured the cash change from the wrong starting point
  ($709.72 instead of $791.02).

---

## 3. Suggestions for further improvement

1. **Ship in-stock items from the warehouse and hold more stock.** Use the
   supplier lead time only for the restocked shortfall, and ship lines
   already in stock in 0–1 days. Replace the "top up to minimum" rule with an
   order-up-to level (minimum + expected demand), based on demand seen so
   far. This targets the 12 deadline rejections directly. It would also stop
   the same-day restock pattern seen on cardstock and glossy paper. Also,
   route every restock through the deadline check when it is tied to an
   order, which closes the request 18 gap.

2. **Track lost demand and widen the range.** Log every NOT_STOCKED and "not
   in catalog" line to a `lost_demand` table. Give the Business Advisor a tool
   to read it, and let it recommend adding items requested repeatedly
   (standard copy paper, poster paper, matte paper, construction paper and
   napkins in this run). Adding an item means inserting it into the
   `inventory` table so `generate_financial_report` values it correctly.
   In this run, nearly half of all requested lines were lost to range alone.

3. **Build customer replies from structured data.** Have Sales return
   structured line outcomes, render the reply from a fixed template (items,
   prices, discount reason, total, delivery date, reason per unsold line),
   and let the model write only the greeting and closing. This would remove
   leaked details, invented offers and mislabeled reasons by construction,
   instead of catching them afterwards.

4. **Make catalog matching consistent.** Add a small synonym table (e.g.
   "printer paper" → A4 paper, "poster board" → Large poster paper) checked
   before the TF-IDF match, so the same wording always maps the same way.

---

## 4. Submission contents

| File | Contents |
| :--- | :--- |
| `project_starter.py` | Complete implementation: tools, five agents, orchestration and the evaluation harness |
| `Agent Workflow.svg`, `Agent Workflow Sequence.svg` | Updated agent workflow diagrams |
| `Agent Workflow.md` | Diagram sources, agent responsibilities, full tool reference, guardrails |
| `test_results.csv` | Evaluation output for all 20 sample requests |
| `business_advisor_report.txt` | Business Advisor notes per request and the end-of-run review |
| `Reflection Report.md` | This report |
