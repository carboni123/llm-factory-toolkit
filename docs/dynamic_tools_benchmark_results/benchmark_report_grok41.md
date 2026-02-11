# Dynamic Tool Calling Benchmark Report

**Model:** `xai/grok-4-1-fast-non-reasoning`
**Date:** 2026-02-11 13:12:04 UTC
**Cases:** 10

## Summary

| Metric | Value |
|--------|-------|
| Pass | 7 |
| Partial | 3 |
| Fail | 0 |
| Error | 0 |
| Total time | 110645ms |
| Total tokens | 116681 |

## Results

| Status | Case | Protocol | Loading | Usage | Overall | Calls | Meta% | Eff% | Ceiling | Time | Tokens |
|--------|------|----------|---------|-------|---------|-------|-------|------|---------|------|--------|
| PASS | crm_summary | 2/2 | 1/1 | 1/1 | 4/4 | 8 | 88% | 12% | - | 11592ms | 8677 |
| PART | task_creation | 2/2 | 1/1 | 0/1 | 3/4 | 7 | 100% | 0% | - | 9452ms | 7168 |
| PASS | calendar_booking | 2/2 | 1/1 | 1/1 | 4/4 | 8 | 75% | 25% | - | 12244ms | 13363 |
| PASS | customer_lookup | 2/2 | 2/2 | 2/2 | 6/6 | 5 | 80% | 20% | - | 11155ms | 8122 |
| PASS | deal_creation | 2/2 | 1/1 | 1/1 | 4/4 | 12 | 75% | 25% | - | 15872ms | 18303 |
| PASS | cross_category | 2/2 | 2/2 | 2/2 | 6/6 | 14 | 71% | 29% | - | 14991ms | 23118 |
| PASS | multi_tool_load | 2/2 | 3/3 | 3/3 | 8/8 | 11 | 73% | 27% | - | 14912ms | 23042 |
| PART | category_browse | 2/2 | 1/1 | 0/1 | 3/4 | 2 | 100% | 0% | - | 5490ms | 3668 |
| PART | group_load | 1/1 | 1/1 | 0/1 | 2/3 | 3 | 100% | 0% | - | 5426ms | 4254 |
| PASS | session_persistence | 2/2 | 1/1 | 1/1 | 4/4 | 4 | 50% | 50% | - | 9511ms | 6966 |

## Failure Details

### task_creation [PARTIAL]

- **Tags:** smoke
- **Missing called:** ['create_task']
- **Response preview:** To create the task, I need the assignee_id (UUID), customer_id (UUID for Joao Santos), and deal_id (UUID or null). Could you provide those details? If Joao Santos is the customer, do you have their customer ID? Also, confirm the assignee.   Once I have them, I can proceed.

### category_browse [PARTIAL]

- **Tags:** protocol
- **Missing called:** ['create_calendar_event']
- **Response preview:** To create the "Team Meeting" event, I need a few more details as required by the tool:  - **Description** (e.g., agenda or notes)? - **Location** (e.g., Zoom link, office room, or "virtual")? - **Customer ID** (UUID for the associated customer; pass null if not applicable)?  "Tomorrow at 10am for 1 

### group_load [PARTIAL]

- **Tags:** protocol
- **Missing called:** ['query_calendar']
- **Response preview:** To check your schedule for 2026-02-10, I need your customer ID (UUID). Could you provide it? Once I have it, I can query the calendar.

## Efficiency Analysis

| Case | Calls | Meta | Business | Overhead% | Efficiency% | Redundant Browses | Wasted Loads | Ceiling |
|------|-------|------|----------|-----------|-------------|-------------------|--------------|---------|
| crm_summary | 8 | 7 | 1 | 88% | 12% | 4 | query_customers | - |
| task_creation | 7 | 7 | 0 | 100% | 0% | 5 | create_task | - |
| calendar_booking | 8 | 6 | 2 | 75% | 25% | 3 | - | - |
| customer_lookup | 5 | 4 | 1 | 80% | 20% | 2 | - | - |
| deal_creation | 12 | 9 | 3 | 75% | 25% | 5 | get_crm_summary | - |
| cross_category | 14 | 10 | 4 | 71% | 29% | 7 | - | - |
| multi_tool_load | 11 | 8 | 3 | 73% | 27% | 6 | - | - |
| category_browse | 2 | 2 | 0 | 100% | 0% | 0 | create_calendar_event | - |
| group_load | 3 | 3 | 0 | 100% | 0% | 1 | create_calendar_event, delete_calendar_event, query_calendar, update_calendar_event | - |
| session_persistence | 4 | 2 | 2 | 50% | 50% | 0 | - | - |

**Aggregate:** 74 calls (58 meta + 16 business) | Ceiling hits: 0 | Redundant browses: 33 | Wasted loads: 8

## Per-Case Details

### crm_summary

- **Status:** pass
- **Duration:** 11592ms
- **Tokens:** 8677
- **Tool calls:** 8 (7 meta + 1 business)
- **Meta overhead:** 88% | **Efficiency:** 12%
- **Hit ceiling:** False
- **Redundant browses:** 4
- **Wasted loads:** ['query_customers']
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['get_crm_summary'], actual=['get_crm_summary', 'query_customers']
- **Tools called:** expected=['get_crm_summary'], actual=['get_crm_summary']
- **Response:** We have **156 total customers** in the CRM.  ### Full CRM Overview: - **Customers**: 156 total     - Active: 142     - Lead: 8     - Prospect: 4     - Inactive: 2    - **Deals**: $127,500.00 open pipe...

### task_creation

- **Status:** partial
- **Duration:** 9452ms
- **Tokens:** 7168
- **Tool calls:** 7 (7 meta + 0 business)
- **Meta overhead:** 100% | **Efficiency:** 0%
- **Hit ceiling:** False
- **Redundant browses:** 5
- **Wasted loads:** ['create_task']
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_task'], actual=['create_task']
- **Tools called:** expected=['create_task'], actual=[]
- **Response:** To create the task, I need the assignee_id (UUID), customer_id (UUID for Joao Santos), and deal_id (UUID or null). Could you provide those details? If Joao Santos is the customer, do you have their cu...

### calendar_booking

- **Status:** pass
- **Duration:** 12244ms
- **Tokens:** 13363
- **Tool calls:** 8 (6 meta + 2 business)
- **Meta overhead:** 75% | **Efficiency:** 25%
- **Hit ceiling:** False
- **Redundant browses:** 3
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_calendar_event'], actual=['create_calendar_event', 'query_calendar']
- **Tools called:** expected=['create_calendar_event'], actual=['query_calendar', 'create_calendar_event']
- **Response:** Your calendar on 2026-02-10 shows the following events (none conflict with 2pm-3pm):  - Haircut - Maria Silva | 09:00 - 10:00 - Beard Trim - Joao Santos | 10:30 - 11:00 - Hair Color - Ana Oliveira | 1...

### customer_lookup

- **Status:** pass
- **Duration:** 11155ms
- **Tokens:** 8122
- **Tool calls:** 5 (4 meta + 1 business)
- **Meta overhead:** 80% | **Efficiency:** 20%
- **Hit ceiling:** False
- **Redundant browses:** 2
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_customers', 'get_customer_context'], actual=['query_customers']
- **Tools called:** expected=['query_customers', 'get_customer_context'], actual=['query_customers']
- **Response:** **Maria Silva** is the customer with phone number **+5511999998888**. Their status is **Active**....

### deal_creation

- **Status:** pass
- **Duration:** 15872ms
- **Tokens:** 18303
- **Tool calls:** 12 (9 meta + 3 business)
- **Meta overhead:** 75% | **Efficiency:** 25%
- **Hit ceiling:** False
- **Redundant browses:** 5
- **Wasted loads:** ['get_crm_summary']
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_deal'], actual=['create_customer', 'create_deal', 'get_crm_summary', 'query_customers']
- **Tools called:** expected=['create_deal'], actual=['query_customers', 'create_customer', 'create_deal']
- **Response:** Deal "Enterprise Plan for Construtora ABC" successfully created (ID: d8642186-39e4-4b28-ae75-3f4960ed7232) with value R$25,000 at stage **Proposal**.  Customer "Construtora ABC" was also created (ID: ...

### cross_category

- **Status:** pass
- **Duration:** 14991ms
- **Tokens:** 23118
- **Tool calls:** 14 (10 meta + 4 business)
- **Meta overhead:** 71% | **Efficiency:** 29%
- **Hit ceiling:** False
- **Redundant browses:** 7
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_calendar', 'create_task'], actual=['create_task', 'query_calendar']
- **Tools called:** expected=['query_calendar', 'create_task'], actual=['query_calendar', 'create_task']
- **Response:** Checked calendar for 2026-02-06: Found 3 appointments.  Created Medium priority follow-up tasks due 2026-02-13 for each:  - Maria Silva (Haircut): Task ID b53bc77d-46b5-465f-b8bd-1041492ead53 - Joao S...

### multi_tool_load

- **Status:** pass
- **Duration:** 14912ms
- **Tokens:** 23042
- **Tool calls:** 11 (8 meta + 3 business)
- **Meta overhead:** 73% | **Efficiency:** 27%
- **Hit ceiling:** False
- **Redundant browses:** 6
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_customer', 'create_deal', 'create_task']
- **Tools called:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_customer', 'create_deal', 'create_task']
- **Response:** Customer **Ana Oliveira** created (ID: `86213b03-bc39-41c9-954b-b03c1d8296ea`).  Deal **Premium Package** created (ID: `dfdbe040-17a8-4e88-9ce8-5c3368d52150`, $5000).  Task **Send proposal** created (...

### category_browse

- **Status:** partial
- **Duration:** 5490ms
- **Tokens:** 3668
- **Tool calls:** 2 (2 meta + 0 business)
- **Meta overhead:** 100% | **Efficiency:** 0%
- **Hit ceiling:** False
- **Wasted loads:** ['create_calendar_event']
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Tools called:** expected=['create_calendar_event'], actual=[]
- **Response:** To create the "Team Meeting" event, I need a few more details as required by the tool:  - **Description** (e.g., agenda or notes)? - **Location** (e.g., Zoom link, office room, or "virtual")? - **Cust...

### group_load

- **Status:** partial
- **Duration:** 5426ms
- **Tokens:** 4254
- **Tool calls:** 3 (3 meta + 0 business)
- **Meta overhead:** 100% | **Efficiency:** 0%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Wasted loads:** ['create_calendar_event', 'delete_calendar_event', 'query_calendar', 'update_calendar_event']
- **Meta-calls:** expected=['browse_toolkit'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_calendar'], actual=['create_calendar_event', 'delete_calendar_event', 'query_calendar', 'update_calendar_event']
- **Tools called:** expected=['query_calendar'], actual=[]
- **Response:** To check your schedule for 2026-02-10, I need your customer ID (UUID). Could you provide it? Once I have it, I can query the calendar....

### session_persistence

- **Status:** pass
- **Duration:** 9511ms
- **Tokens:** 6966
- **Tool calls:** 4 (2 meta + 2 business)
- **Meta overhead:** 50% | **Efficiency:** 50%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['get_weather'], actual=['get_weather']
- **Tools called:** expected=['get_weather'], actual=['get_weather']
- **Response:** The weather in Tokyo is currently sunny with a temperature of 22°C....
