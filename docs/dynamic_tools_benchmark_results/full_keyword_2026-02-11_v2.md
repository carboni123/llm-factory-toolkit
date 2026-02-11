# Dynamic Tool Calling Benchmark Report

**Model:** `openai/gpt-4o-mini`
**Date:** 2026-02-11 19:14:37 UTC
**Cases:** 13

## Summary

| Metric | Value |
|--------|-------|
| Pass | 10 |
| Partial | 3 |
| Fail | 0 |
| Error | 0 |
| Total time | 140243ms |
| Total tokens | 154817 |

## Results

| Status | Case | Protocol | Loading | Usage | Response | Overall | Calls | Meta% | Eff% | Ceiling | Time | Tokens |
|--------|------|----------|---------|-------|----------|---------|-------|-------|------|---------|------|--------|
| PASS | crm_summary | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 3 | 67% | 33% | - | 8646ms | 4431 |
| PASS | task_creation | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 4 | 75% | 25% | - | 9502ms | 5831 |
| PASS | calendar_booking | 2/2 | 2/2 | 2/2 | 2/2 | 8/8 | 5 | 60% | 40% | - | 10059ms | 7937 |
| PASS | customer_lookup | 2/2 | 1/1 | 1/1 | n/a | 4/4 | 6 | 83% | 17% | - | 8261ms | 8509 |
| PART | deal_creation | 2/2 | 0/1 | 0/1 | 1/1 | 3/5 | 7 | 86% | 14% | - | 12763ms | 9532 |
| PART | cross_category | 2/2 | 1/2 | 1/2 | n/a | 4/6 | 5 | 60% | 40% | - | 9194ms | 8247 |
| PASS | multi_tool_load | 2/2 | 3/3 | 3/3 | n/a | 8/8 | 7 | 57% | 43% | - | 10422ms | 8424 |
| PASS | category_browse | 2/2 | 1/1 | 1/1 | n/a | 4/4 | 3 | 67% | 33% | - | 6063ms | 4715 |
| PASS | group_load | 1/1 | 1/1 | 1/1 | n/a | 3/3 | 3 | 67% | 33% | - | 5379ms | 5508 |
| PASS | customer_update | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 5 | 80% | 20% | - | 7257ms | 6209 |
| PART | deal_lifecycle | 2/2 | 0/2 | 0/2 | n/a | 2/6 | 29 | 17% | 83% | - | 35089ms | 71343 |
| PASS | task_cleanup | 2/2 | 2/2 | 2/2 | n/a | 6/6 | 8 | 62% | 38% | - | 11175ms | 7733 |
| PASS | session_persistence | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 4 | 50% | 50% | - | 6433ms | 6398 |

## Failure Details

### deal_creation [PARTIAL]

- **Tags:** smoke
- **Missing loaded:** ['create_deal']
- **Missing called:** ['create_deal']
- **Response preview:** It seems that I currently don't have the specific tool to register a new deal directly. However, I can provide you with the CRM summary, which includes the current state of deals and tasks.  ### CRM Summary: - **Total Customers:** 156   - Active: 142   - Lead: 8   - Prospect: 4   - Inactive: 2  - **

### cross_category [PARTIAL]

- **Tags:** cross-category
- **Missing loaded:** ['query_calendar']
- **Missing called:** ['query_calendar']
- **Response preview:** I created a follow-up task for the customer regarding their appointment on **2026-02-06**. The task is due on **2026-02-13** with **Medium** priority.   If you need to create more tasks or have any other requests, let me know!

### deal_lifecycle [PARTIAL]

- **Tags:** multi-tool
- **Missing loaded:** ['query_deals', 'ANY(update_deal | delete_deal)']
- **Missing called:** ['query_deals', 'ANY(update_deal | delete_deal)']
- **Response preview:** [Tool executions completed without a final assistant message. Review returned payloads for actionable results.]

## Efficiency Analysis

| Case | Calls | Meta | Business | Overhead% | Efficiency% | Redundant Browses | Wasted Loads | Ceiling |
|------|-------|------|----------|-----------|-------------|-------------------|--------------|---------|
| crm_summary | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| task_creation | 4 | 3 | 1 | 75% | 25% | 1 | - | - |
| calendar_booking | 5 | 3 | 2 | 60% | 40% | 1 | - | - |
| customer_lookup | 6 | 5 | 1 | 83% | 17% | 3 | - | - |
| deal_creation | 7 | 6 | 1 | 86% | 14% | 4 | get_customer_context | - |
| cross_category | 5 | 3 | 2 | 60% | 40% | 1 | - | - |
| multi_tool_load | 7 | 4 | 3 | 57% | 43% | 2 | - | - |
| category_browse | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| group_load | 3 | 2 | 1 | 67% | 33% | 0 | create_calendar_event, delete_calendar_event, update_calendar_event | - |
| customer_update | 5 | 4 | 1 | 80% | 20% | 2 | - | - |
| deal_lifecycle | 29 | 5 | 24 | 17% | 83% | 3 | - | - |
| task_cleanup | 8 | 5 | 3 | 62% | 38% | 3 | - | - |
| session_persistence | 4 | 2 | 2 | 50% | 50% | 0 | - | - |

**Aggregate:** 89 calls (46 meta + 43 business) | Ceiling hits: 0 | Redundant browses: 20 | Wasted loads: 4

## Per-Case Details

### crm_summary

- **Status:** pass
- **Duration:** 8646ms
- **Tokens:** 4431
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['get_crm_summary'], actual=['get_crm_summary']
- **Tools called:** expected=['get_crm_summary'], actual=['get_crm_summary']
- **Response content:** score=1/1, missing=[]
- **Response:** Here's the CRM overview:  ### Customers - **Total Customers:** 156   - **Active:** 142   - **Lead:** 8   - **Prospect:** 4   - **Inactive:** 2  ### Deals - **Open Pipeline Value:** $127,500.00   - **T...

### task_creation

- **Status:** pass
- **Duration:** 9502ms
- **Tokens:** 5831
- **Tool calls:** 4 (3 meta + 1 business)
- **Meta overhead:** 75% | **Efficiency:** 25%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_task'], actual=['create_task']
- **Tools called:** expected=['create_task'], actual=['create_task']
- **Response content:** score=1/1, missing=[]
- **Response:** The follow-up task to call Joao Santos has been created successfully:  - **Title:** Call Joao Santos - **Due Date:** 2026-02-14 - **Priority:** High  If you need anything else, feel free to ask!...

### calendar_booking

- **Status:** pass
- **Duration:** 10059ms
- **Tokens:** 7937
- **Tool calls:** 5 (3 meta + 2 business)
- **Meta overhead:** 60% | **Efficiency:** 40%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_calendar', 'create_calendar_event'], actual=['create_calendar_event', 'query_calendar']
- **Tools called:** expected=['query_calendar', 'create_calendar_event'], actual=['query_calendar', 'create_calendar_event']
- **Response content:** score=2/2, missing=[]
- **Response:** The haircut appointment for Maria Silva has been successfully booked for **February 10, 2026, at 2:00 PM**. Here are the details:  - **Title:** Haircut Appointment for Maria Silva - **Start Time:** 2:...

### customer_lookup

- **Status:** pass
- **Duration:** 8261ms
- **Tokens:** 8509
- **Tool calls:** 6 (5 meta + 1 business)
- **Meta overhead:** 83% | **Efficiency:** 17%
- **Hit ceiling:** False
- **Redundant browses:** 3
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['ANY(query_customers | get_customer_context)'], actual=['query_customers']
- **Tools called:** expected=['ANY(query_customers | get_customer_context)'], actual=['query_customers']
- **Response content:** score=n/a, missing=[]
- **Response:** The customer with the phone number **+5511999998888** is:  - **Name:** Maria Silva - **Status:** Active...

### deal_creation

- **Status:** partial
- **Duration:** 12763ms
- **Tokens:** 9532
- **Tool calls:** 7 (6 meta + 1 business)
- **Meta overhead:** 86% | **Efficiency:** 14%
- **Hit ceiling:** False
- **Redundant browses:** 4
- **Wasted loads:** ['get_customer_context']
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_deal'], actual=['get_crm_summary', 'get_customer_context']
- **Tools called:** expected=['create_deal'], actual=['get_crm_summary']
- **Response content:** score=1/1, missing=[]
- **Response:** It seems that I currently don't have the specific tool to register a new deal directly. However, I can provide you with the CRM summary, which includes the current state of deals and tasks.  ### CRM S...

### cross_category

- **Status:** partial
- **Duration:** 9194ms
- **Tokens:** 8247
- **Tool calls:** 5 (3 meta + 2 business)
- **Meta overhead:** 60% | **Efficiency:** 40%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_calendar', 'create_task'], actual=['create_task', 'query_tasks']
- **Tools called:** expected=['query_calendar', 'create_task'], actual=['query_tasks', 'create_task']
- **Response content:** score=n/a, missing=[]
- **Response:** I created a follow-up task for the customer regarding their appointment on **2026-02-06**. The task is due on **2026-02-13** with **Medium** priority.   If you need to create more tasks or have any ot...

### multi_tool_load

- **Status:** pass
- **Duration:** 10422ms
- **Tokens:** 8424
- **Tool calls:** 7 (4 meta + 3 business)
- **Meta overhead:** 57% | **Efficiency:** 43%
- **Hit ceiling:** False
- **Redundant browses:** 2
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_customer', 'create_deal', 'create_task']
- **Tools called:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_customer', 'create_deal', 'create_task']
- **Response content:** score=n/a, missing=[]
- **Response:** All tasks have been successfully completed:  1. **Customer Registered**: Ana Oliveira (ID: **9b65b452-a5f3-4700-9855-eecfa643b46a**) 2. **Deal Created**: Premium Package for $5000 (ID: **f09e69fe-c319...

### category_browse

- **Status:** pass
- **Duration:** 6063ms
- **Tokens:** 4715
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Tools called:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Response content:** score=n/a, missing=[]
- **Response:** The Team Meeting has been successfully scheduled for tomorrow at 10 AM for 1 hour. Here are the details:  - **Title:** Team Meeting - **Start Time:** October 10, 2023, at 10:00 AM - **Duration:** 60 m...

### group_load

- **Status:** pass
- **Duration:** 5379ms
- **Tokens:** 5508
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Wasted loads:** ['create_calendar_event', 'delete_calendar_event', 'update_calendar_event']
- **Meta-calls:** expected=['browse_toolkit'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['ANY(query_calendar)'], actual=['create_calendar_event', 'delete_calendar_event', 'query_calendar', 'update_calendar_event']
- **Tools called:** expected=['ANY(query_calendar)'], actual=['query_calendar']
- **Response content:** score=n/a, missing=[]
- **Response:** On **February 10, 2026**, your schedule includes the following events:  1. **Haircut** with Maria Silva: **09:00 - 10:00** 2. **Beard Trim** with Joao Santos: **10:30 - 11:00** 3. **Hair Color** with ...

### customer_update

- **Status:** pass
- **Duration:** 7257ms
- **Tokens:** 6209
- **Tool calls:** 5 (4 meta + 1 business)
- **Meta overhead:** 80% | **Efficiency:** 20%
- **Hit ceiling:** False
- **Redundant browses:** 2
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['update_customer'], actual=['update_customer']
- **Tools called:** expected=['update_customer'], actual=['update_customer']
- **Response content:** score=1/1, missing=[]
- **Response:** Maria Silva's email address has been successfully updated to **maria.silva@newco.com**. If you need any further assistance, feel free to ask!...

### deal_lifecycle

- **Status:** partial
- **Duration:** 35089ms
- **Tokens:** 71343
- **Tool calls:** 29 (5 meta + 24 business)
- **Meta overhead:** 17% | **Efficiency:** 83%
- **Hit ceiling:** False
- **Redundant browses:** 3
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_deals', 'ANY(update_deal | delete_deal)'], actual=['get_crm_summary', 'get_customer_context']
- **Tools called:** expected=['query_deals', 'ANY(update_deal | delete_deal)'], actual=['get_crm_summary', 'get_customer_context']
- **Response content:** score=n/a, missing=[]
- **Response:** [Tool executions completed without a final assistant message. Review returned payloads for actionable results.]...

### task_cleanup

- **Status:** pass
- **Duration:** 11175ms
- **Tokens:** 7733
- **Tool calls:** 8 (5 meta + 3 business)
- **Meta overhead:** 62% | **Efficiency:** 38%
- **Hit ceiling:** False
- **Redundant browses:** 3
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_tasks', 'ANY(delete_task | update_task)'], actual=['delete_task', 'query_tasks', 'update_task']
- **Tools called:** expected=['query_tasks', 'ANY(delete_task | update_task)'], actual=['query_tasks', 'delete_task', 'update_task']
- **Response content:** score=n/a, missing=[]
- **Response:** Here are the updates:  - **Overdue Tasks**:    - 12 overdue tasks were found, including:     1. **Call back client re: invoice** - Due: 2026-01-28 [HIGH]     2. **Send contract revision** - Due: 2026-...

### session_persistence

- **Status:** pass
- **Duration:** 6433ms
- **Tokens:** 6398
- **Tool calls:** 4 (2 meta + 2 business)
- **Meta overhead:** 50% | **Efficiency:** 50%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['get_weather'], actual=['get_weather']
- **Tools called:** expected=['get_weather'], actual=['get_weather']
- **Response content:** score=1/1, missing=[]
- **Response:** The weather in Tokyo is currently sunny, with a temperature of 22°C....
