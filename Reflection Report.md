# Reflection Report: Beaver's Choice Multi-Agent System

Multi-agent quoting, inventory and order-fulfillment system for the Beaver's
Choice Paper Company, built with **smolagents** and evaluated on the 20
requests in `quote_requests_sample.csv`.

## Results at a glance

| Rubric requirement | Result |
| :--- | :--- |
| At least 3 requests change the cash balance | **11 of 20** (all increases) |
| At least 3 quote requests successfully fulfilled | **11 of 20** sold at least one line; 3 fully fulfilled |
| Not every request fulfilled, with reasons given | **9 of 20** sold nothing; all 42 unsold lines have a stated reason |
| Every helper function used in a tool | All 7 (table in section 1.5) |
| Maximum of five agents | 5: Orchestrator + 4 workers |

Over the run, revenue was **$2,001.62** from 18 order lines (16,150 units).
Total assets rose from **$50,000.00 to $50,294.12**. That increase equals the
gross profit exactly, and no sale was made below cost. `validate_results.py`
passes every check on this run (section 2.4).

---

## 1. The multi-agent system

### 1.1 Architecture

![Agent workflow diagram](Agent%20Workflow.svg)

Files: `Agent Workflow.svg` (architecture), `Agent Workflow Sequence.svg`
(step-by-step flow for one request) and `Agent Workflow.md` (both diagrams as
Mermaid, plus the full tool reference).

| Agent | Responsibility | Tools | Reports back as |
| :--- | :--- | :--- | :--- |
| **Orchestrator** | Maps the customer's items to exact catalog names, delegates to Inventory, Quoting and Sales in order, decides each item's outcome | `match_catalog_items_tool` | JSON plan: outcome + reason code per item, short note |
| **Inventory Agent** | Decides whether each item can be supplied; restocks from the supplier when cash and lead time allow | `check_stock_tool`, `inventory_snapshot_tool`, `check_cash_tool`, `supplier_lead_time_tool`, `reorder_stock_tool` | JSON: status + reason code per item (no stock counts) |
| **Quoting Agent** | Prices available items with list prices and bulk discounts, guided by quote history | `quote_history_tool`, `price_lookup_tool`, `calculate_quote_tool` | JSON: quote lines, discount rate, total, rationale |
| **Sales Agent** | Checks each line can be delivered by the customer's date and records the sale | `delivery_check_tool`, `record_sale_tool` | JSON: SOLD / REJECTED per line, delivery date or reason |
| **Business Advisor** | Read-only analyst: reviews cash, stock and lost demand after every reply and at the end of the run | `financial_report_tool`, `cash_trend_tool`, `stock_health_tool` | Internal notes in `business_advisor_report.txt` |

Responsibilities do not overlap. Only Inventory buys stock, only Sales records
sales, only Quoting sets prices, and the Advisor writes nothing to the
database. The Orchestrator has no database tools at all, and it does not write
the customer reply.

### 1.2 How a request flows

1. The orchestration layer (`call_your_multi_agent_system`) parses the
   **request date** and the **deliver-by date** from the text in code. It
   stores them in a per-request context that every database tool reads.
2. The **Orchestrator** lists the items and calls `match_catalog_items_tool`,
   a TF-IDF character n-gram match over the 46-item catalog. Candidates of a
   different paper size are removed (an A3 request never matches a
   standard-size paper). The Orchestrator picks the candidate that is
   genuinely the same product, or marks the item "not in our catalog".
3. **Inventory** checks each matched item: NOT_STOCKED if the company does
   not carry it, AVAILABLE if stock covers the order. If stock is short, it
   checks supplier lead time and cash, and restocks only if the delivery
   arrives by the customer's date.
4. **Quoting** prices the available items. List price is supplier cost plus
   a 25% markup. Bulk discounts are 5% from 1,000 units, 10% from 5,000 and
   15% from 10,000, with a price floor of cost plus 5%.
5. **Sales** re-checks the deadline and stock for each line, then records it
   with `create_transaction('sales')`.
6. The **Orchestrator** returns a JSON plan: for every requested item, SOLD or
   NOT_SUPPLIED with a reason code, plus a one- or two-sentence personal note.
7. The orchestration layer **renders the customer reply in code**. Prices,
   the discount rationale, the total and delivery dates come from the
   recorded sales; each unsold item gets a plain-language reason. A final
   sanitizer (`sanitize_customer_reply`) removes any sentence that matches
   `CUSTOMER_BLOCKLIST`.
8. Every unsold item is logged with its reason code and quantity (lost
   demand). The **Business Advisor** reviews the outcome and the lost-demand
   log and appends recommendations to `business_advisor_report.txt`. These
   notes are internal and are never sent to the customer.

### 1.3 Design decisions and why

**Five agents, split by business function.** The split mirrors how a small
distributor works: customer service, stock control, pricing and order
processing are separate jobs. Keeping them apart means each agent has a short
prompt and only the tools it needs. That matters for a small model: fewer
tools per agent meant fewer wrong tool choices. The fifth slot went to the
Business Advisor (a "stand out" suggestion) rather than a negotiating
customer agent, which would have been a sixth agent.

**Structured hand-offs instead of free text.** In an earlier version each
agent replied in prose, and the Orchestrator wrote the customer reply itself.
The review of that version found replies that leaked a stock count
("399 units available") and offered services the company never performs
(outside sourcing, expedited delivery, substitutions, invoices). Now every
agent except the Advisor returns a fixed JSON object, and a
`final_answer_checks` function compares it with what the tools recorded:

- extra fields (stock counts, costs, cash) are rejected, so internal data
  never reaches the Orchestrator;
- figures that differ from the records are sent back to the agent to fix;
- the only model-written text the customer sees is the Orchestrator's short
  note, which is rejected if it contains figures, internal details or offers.

**The Advisor runs after the reply, not as an Orchestrator tool.** If the
Orchestrator could call the Advisor, its internal notes could leak into the
customer reply. Running it afterwards keeps those notes out of every reply.

**Rules that matter are enforced in code, not prompts.** Test runs showed
failures that prompting alone did not fix. Each became a code-level
guardrail:

| Failure seen in testing | Guardrail |
| :--- | :--- |
| Agents dropped or invented dates (one used 2023-10-03), so stock and cash read as $0 | Dates parsed in code; database tools take no date arguments |
| Customer wording ("A4 glossy paper") sent to Inventory instead of catalog names | Tools reject non-catalog names and list the closest exact names |
| A sale recorded after its own delivery check said it would be late | `record_sale_tool` re-checks the deadline itself |
| Orchestrator skipped Sales, then told the customer they were charged | `plan_matches_records`: available items must reach Sales; SOLD items must be recorded sales |
| Reply totals or delivery dates differed from the database | Reply rendered in code from the recorded sales |
| Replies leaked a stock count and offered unperformed services | JSON reports with no internal fields; note checked for offers; `sanitize_customer_reply` as a final net |
| "A3 glossy paper" and "A5 colored paper" sold as standard-size paper | `match_catalog_items_tool` drops candidates of a different paper size |
| Repeated delegation could double-book a sale | Sales and customer restocks are idempotent within a request |
| Parallel tool calls could race on SQLite | `max_tool_threads=1` on every agent |

**Pricing with a markup.** `init_database` values stock and records purchases
at catalog `unit_price`. Selling at that same price with a bulk discount
would lose money on every restocked sale. A 25% markup with discounts capped
at 15% keeps every sale above cost. The markup is never shown to customers.

**Framework and model.** smolagents was chosen for its built-in managed-agent
delegation (`managed_agents`), `@tool` schemas generated from docstrings, and
`final_answer_checks`, which made the JSON report checks straightforward. It
also connects to the course's OpenAI-compatible proxy without an adapter. The
model is `gpt-5-mini` with `reasoning_effort='low'`: it followed the
multi-step procedure far more reliably than `gpt-4o-mini`, and low effort
kept each request to roughly 25–90 seconds.

### 1.4 Stand-out features

- **Terminal animation.** Every tool call and hand-off emits a status event
  to a live spinner and timeline (`ProgressDisplay`, built on `rich`), so you
  can watch each request move through the agents. Rejected reports show up
  as e.g. `Inventory Agent: report did not match the records - redoing it`.
- **Business Advisor with lost-demand evidence.** After each reply the Advisor
  receives the request, the reply and that request's unsold items with reason
  codes. At the end of the run it receives the lost-demand log grouped by
  item and reason, which is also written to `business_advisor_report.txt`.
  Its recommendations now cite those counts (section 2.3).

### 1.5 Helper function coverage

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
| 4 | 04-04 | 1 / 3 | $527.00 | +$33.85 | A3 paper, "printer paper" (not in catalog); A4 restocked 9,863 sheets and sold at 15% off |
| 5 | 04-05 | 2 / 2 | $109.50 | +$74.05 | none |
| 6 | 04-05 | 2 / 3 | $118.90 | +$73.90 | washi tape (not stocked) |
| 7 | 04-06 | 1 / 3 | $37.60 | +$7.60 | construction paper (not stocked); white printer paper (not in catalog) |
| 8 | 04-07 | 3 / 4 | $510.72 | +$428.72 | matte A3 (not in catalog); 5% discount at exactly 1,000 units |
| 9 | 04-07 | 1 / 4 | $125.00 | +$25.00 | matte, recycled (not stocked); A5 colored (not in catalog) |
| 10 | 04-07 | 0 / 3 | $0.00 | $0.00 | A4 cannot arrive by 04-10; A3 glossy (not in catalog); envelopes (not stocked) |
| 11 | 04-08 | 0 / 2 | $0.00 | $0.00 | A4 and cardstock cannot arrive by 04-10 |
| 12 | 04-08 | 1 / 3 | $37.60 | +$7.60 | copy paper, napkins (not stocked) |
| 13 | 04-08 | 2 / 2 | $181.40 | +$36.40 | none |
| 14 | 04-08 | 0 / 2 | $0.00 | $0.00 | A3 glossy (not in catalog); matte (not stocked) |
| 15 | 04-09 | 1 / 3 | $94.00 | +$19.00 | A4 ×5,000 cannot arrive in time; poster paper (not stocked) |
| 16 | 04-12 | 0 / 3 | $0.00 | $0.00 | A4 ×10,000 and "cardboard" (matched to cardstock) cannot arrive in time; A3 (not in catalog) |
| 17 | 04-13 | 0 / 3 | $0.00 | $0.00 | A4 cannot arrive in time; construction paper (not stocked); poster board (not in catalog) |
| 18 | 04-14 | 0 / 5 | $0.00 | $0.00 | A4 cannot arrive in time; plates in stock but a 4-day delivery misses 04-15; A3 (not in catalog); napkins, cups (not stocked) |
| 19 | 04-14 | 0 / 3 | $0.00 | $0.00 | cardstock, colored paper cannot arrive in time; copy paper (not stocked) |
| 20 | 04-15 | 1 / 3 | $178.60 | +$28.60 | glossy ×2,000 cannot arrive in time; A3 matte (not in catalog) |
| 21 | 04-17 | 0 / 3 | $0.00 | $0.00 | flyers, posters (not stocked); tickets (not in catalog) |

### 2.2 Line-level and financial summary

| Outcome | Lines | Share |
| :--- | ---: | ---: |
| Sold | 18 | 30% |
| Not stocked (item is in the catalog but not carried) | 17 | 28% |
| Not in the catalog | 13 | 22% |
| Cannot restock by the customer's date | 11 | 18% |
| In stock, but delivery misses the customer's date | 1 | 2% |
| **Total requested** | **60** | |

| Finance | Amount |
| :--- | ---: |
| Revenue | $2,001.62 |
| Cost of goods sold (at supplier cost) | $1,707.50 |
| Gross profit (14.7% margin) | $294.12 |
| Restock purchases | $1,185.60 (13 orders) |
| Cash | $45,059.70 → $45,875.72 |
| Inventory value | $4,940.30 → $4,418.40 |
| Total assets | $50,000.00 → $50,294.12 |

### 2.3 Strengths

- **Replies match the books, by construction.** Every figure in a reply is
  rendered from the recorded sales. The totals stated across all replies add
  up to $2,001.62, exactly the `sales` revenue in the `transactions` table,
  and every no-sale reply says nothing was charged.
- **Transparent, consistent replies.** All 20 replies follow the same shape:
  what was sold (with the customer's own wording, unit price, discount and
  delivery date), the discount rationale, the total, and a plain reason for
  each of the 42 unsold lines. A customer can see why "A3 glossy paper"
  (request 14) or "paper plates" (request 18) could not be supplied.
- **No internal information or invented services.** No reply contains a
  stock count, cost, supplier detail, agent or tool name, or an offer of
  sourcing, expedited delivery, substitutions or invoices. Request 21, which
  previously leaked "399 units available", now gives only the reasons.
- **Wrong-size sales eliminated.** In the previous run, "A3 glossy paper"
  (twice) and "A5 colored paper" were sold as standard-size paper, a wrongful
  $387.50 in charges. With the size guard, all eight A3/A5 lines in this run
  were correctly declined as not in the catalog.
- **Correct bulk pricing.** Discounts were applied exactly at their tiers:
  15% on 10,000 A4 sheets (request 4) and 5% at exactly 1,000 units
  (requests 8 and 20). Every other reply explains why list prices applied.
- **Deadlines respected.** Requests 10 and 11 ask for delivery within 2–3
  days, but restocking takes 4. Both were declined with the reason, and
  nothing was charged. In request 18 the paper plates were in stock but a
  500-unit delivery takes 4 days, so the line was rejected rather than
  promised late.
- **Profitable and solvent.** Every sale cleared the price floor (the lowest
  line was 5.4% over cost), giving a 14.7% gross margin. All 13 restocks were
  checked against cash first, and no item ended below its minimum level.
- **Advisor recommendations backed by evidence.** The end-of-run review cites
  the lost-demand log directly, e.g. *"Prioritise A4 paper availability:
  6 lost requests totalling 17,200 units (reason: cannot_supply_in_time)"*.

### 2.4 Post-run validation

`validate_results.py` checks the output independently of the agents. On this
run it reports:

- no blocked phrase in any of the 20 replies;
- stated totals equal recorded sales on every date;
- no request where nothing was sold but cash still moved (all 13 restocks
  were for lines that sold);
- rubric minimums met (11 cash changes, 11 fulfilled, 9 not fulfilled).

### 2.5 Areas for improvement

- **Too few items stocked (50% of all lines).** Only 18 of 46 catalog items
  are stocked (fixed by the starter's seeded inventory), and a further 13
  lines asked for items not in the catalog at all. The lost-demand log shows
  repeat demand for poster paper (3 requests, 4,500 units), napkins (2,
  2,100), standard copy paper (2, 1,500), matte paper (2, 1,300) and
  construction paper (2, 700), plus 13,600 units of A3 paper across 7
  requests.
- **Lean stock turns into missed deadlines (20% of all lines).** Restocks
  only bring stock back to the minimum level, so every one of the 13 restocks
  was bought on the same day as a sale of that item, with a 4–7 day lead
  time. From 04-12 onward, against 04-15 deadlines, those lead times rejected
  every order (requests 16–19). A4 paper alone lost 17,200 units this way. In
  addition, `delivery_check_tool` applies the supplier lead time even to
  lines already in stock, which overstates delivery times (request 18's
  plates).
- **Item matching still depends on the model.** "Printer paper" mapped to A4
  paper in request 5, to standard copy paper in request 12, and to nothing in
  requests 4 and 7. "Cardboard for signage" was matched to cardstock (request
  16), so the customer was told it could not arrive in time instead of that
  we do not sell it. "500 reams" was treated as 500 units, as instructed, but
  a ream is 500 sheets.
- **Advisor details.** The end-of-run review measured the cash change from
  the end of 04-01 instead of its start ($734.72 instead of $816.02), because
  `cash_trend_tool` is given the first request date rather than the day
  before. The lost-demand log also groups by the customer's wording when an
  item is not in the catalog, so "A3 matte paper" and "matte A3 paper" are
  counted separately.
- **A guardrail gap in restocking.** A "routine top-up" restock is not
  checked against the customer's deadline. It did not cause a surplus
  purchase in this run, but it did in an earlier run (request 18 bought
  $50 of A4 paper for an order that was then rejected).

---

## 3. Suggestions for further improvement

1. **Ship in-stock items from the warehouse and hold more stock.** Use the
   supplier lead time only for the restocked shortfall, and ship lines
   already in stock in 0–1 days. Replace the "top up to minimum" rule with an
   order-up-to level (minimum + expected demand), set from the demand seen
   so far, starting with A4 paper and cardstock. This targets the 12
   deadline rejections directly and would end the same-day restock pattern.
   Route every restock tied to an order through the deadline check, which
   closes the top-up gap.

2. **Widen the range using the lost-demand log.** The log is now structured,
   so it can drive range decisions: add items requested repeatedly (poster
   paper, napkins, standard copy paper, matte paper, construction paper) by
   inserting them into the `inventory` table so `generate_financial_report`
   values them correctly, and consider an A3 line. Normalize log keys (size +
   base product) so variants of the same request are counted together.
   Half of all requested lines in this run were lost to range alone.

3. **Make catalog matching deterministic.** Extend the size guard into a
   small rules layer checked before the TF-IDF match: a synonym table
   ("printer paper" → A4 paper, "poster board" → Large poster paper), a
   material check (cardboard is not cardstock), and unit conversion (1 ream
   = 500 sheets). The same wording would then always map the same way.

4. **Correct the Advisor's comparison window.** Pass the day before the first
   request as `before_date` for the end-of-run cash trend, so its summary
   matches the financial report.

---

## 4. Response to review feedback

| Review point | Change |
| :--- | :--- |
| Sanitize every customer-facing response (`CUSTOMER_BLOCKLIST`, `sanitize_customer_reply`) | Added. It runs on every reply before it is returned. It removes whole sentences rather than replacing words, so replies stay readable. |
| Audit every row of `test_results.csv` | All 20 rows checked, by hand and by `validate_results.py`: no stock counts, internal details or unperformed services. |
| Render replies from structured data | Done for the Orchestrator and the three worker agents: JSON reports checked against the records, and the reply is rendered in code. |
| Post-run validation script | `validate_results.py`: blocked phrases, totals vs recorded sales, no-sale cash movement, rubric minimums. |
| Connect advisor recommendations to lost demand | Unsold items are logged with reason codes and units. The Advisor receives the log per request and grouped at the end of the run, and cites it. |

---

## 5. Submission contents

| File | Contents |
| :--- | :--- |
| `project_starter.py` | Complete implementation: tools, five agents, JSON report checks, reply renderer and sanitizer, orchestration and the evaluation harness |
| `validate_results.py` | Post-run validation of `test_results.csv` against the database |
| `Agent Workflow.svg`, `Agent Workflow Sequence.svg` | Agent workflow diagrams |
| `Agent Workflow.md` | Diagram sources, agent responsibilities and JSON reports, full tool reference, guardrails |
| `test_results.csv` | Evaluation output for all 20 sample requests |
| `business_advisor_report.txt` | Business Advisor notes per request, the lost-demand log and the end-of-run review |
| `Reflection Report.md` | This report |
