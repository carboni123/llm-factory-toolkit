# Dynamic Tool Calling Benchmark Report

**Model:** `openai/gpt-4o-mini`
**Date:** 2026-02-11 19:15:02 UTC
**Cases:** 13

## Summary

| Metric | Value |
|--------|-------|
| Pass | 10 |
| Partial | 2 |
| Fail | 1 |
| Error | 0 |
| Total time | 163044ms |
| Total tokens | 90341 |

## Results

| Status | Case | Protocol | Loading | Usage | Response | Overall | Calls | Meta% | Eff% | Ceiling | Time | Tokens |
|--------|------|----------|---------|-------|----------|---------|-------|-------|------|---------|------|--------|
| PASS | crm_summary | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 3 | 67% | 33% | - | 9507ms | 2509 |
| PASS | task_creation | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 3 | 67% | 33% | - | 8814ms | 2844 |
| PART | calendar_booking | 2/2 | 1/2 | 1/2 | 2/2 | 6/8 | 3 | 67% | 33% | - | 8078ms | 2681 |
| PASS | customer_lookup | 2/2 | 1/1 | 1/1 | n/a | 4/4 | 3 | 67% | 33% | - | 6588ms | 3071 |
| PASS | deal_creation | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 3 | 67% | 33% | - | 6418ms | 2485 |
| PASS | cross_category | 2/2 | 2/2 | 2/2 | n/a | 6/6 | 6 | 33% | 67% | - | 15047ms | 5273 |
| FAIL | multi_tool_load | 1/2 | 2/3 | 2/3 | n/a | 5/8 | 5 | 40% | 60% | - | 12022ms | 5155 |
| PASS | category_browse | 2/2 | 1/1 | 1/1 | n/a | 4/4 | 3 | 67% | 33% | - | 6706ms | 3644 |
| PASS | group_load | 1/1 | 1/1 | 1/1 | n/a | 3/3 | 3 | 67% | 33% | - | 7994ms | 4051 |
| PASS | customer_update | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 3 | 67% | 33% | - | 6644ms | 2637 |
| PART | deal_lifecycle | 2/2 | 1/2 | 1/2 | n/a | 4/6 | 40 | 5% | 95% | - | 53431ms | 48516 |
| PASS | task_cleanup | 2/2 | 2/2 | 2/2 | n/a | 6/6 | 7 | 57% | 43% | - | 13052ms | 4765 |
| PASS | session_persistence | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 4 | 50% | 50% | - | 8743ms | 2710 |

## Failure Details

### calendar_booking [PARTIAL]

- **Tags:** multi-tool
- **Missing loaded:** ['create_calendar_event']
- **Missing called:** ['create_calendar_event']
- **Response preview:** On February 10, 2026, the following appointments are scheduled:  1. **Haircut** - Maria Silva | 09:00 - 10:00 2. **Beard Trim** - Joao Santos | 10:30 - 11:00 3. **Hair Color** - Ana Oliveira | 14:00 - 15:30  Since Maria already has a haircut appointment from 09:00 to 10:00 and another appointment ov

### multi_tool_load [FAIL]

- **Tags:** multi-tool
- **Missing meta-calls:** ['find_tools']
- **Missing loaded:** ['create_customer']
- **Missing called:** ['create_customer']
- **Response preview:** Here's what I've accomplished:  1. **Customer Registration**: A task has been created to register Ana Oliveira as a new customer.    - **Task ID**: d15ec720-1131-459b-9eee-83d7c187cb34    - **Due Date**: 2023-10-10    - **Priority**: High  2. **Deal Creation**: The deal "Premium Package" for $5000 h

### deal_lifecycle [PARTIAL]

- **Tags:** multi-tool
- **Missing loaded:** ['ANY(update_deal | delete_deal)']
- **Missing called:** ['ANY(update_deal | delete_deal)']
- **Response preview:** [Tool executions completed without a final assistant message. Review returned payloads for actionable results.]

## Efficiency Analysis

| Case | Calls | Meta | Business | Overhead% | Efficiency% | Redundant Browses | Wasted Loads | Ceiling |
|------|-------|------|----------|-----------|-------------|-------------------|--------------|---------|
| crm_summary | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| task_creation | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| calendar_booking | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| customer_lookup | 3 | 2 | 1 | 67% | 33% | 0 | get_customer_context | - |
| deal_creation | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| cross_category | 6 | 2 | 4 | 33% | 67% | 0 | - | - |
| multi_tool_load | 5 | 2 | 3 | 40% | 60% | 0 | - | - |
| category_browse | 3 | 2 | 1 | 67% | 33% | 0 | query_calendar | - |
| group_load | 3 | 2 | 1 | 67% | 33% | 0 | create_calendar_event, delete_calendar_event, update_calendar_event | - |
| customer_update | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| deal_lifecycle | 40 | 2 | 38 | 5% | 95% | 0 | - | - |
| task_cleanup | 7 | 4 | 3 | 57% | 43% | 2 | - | - |
| session_persistence | 4 | 2 | 2 | 50% | 50% | 0 | - | - |

**Aggregate:** 86 calls (28 meta + 58 business) | Ceiling hits: 0 | Redundant browses: 2 | Wasted loads: 5

## Per-Case Details

### crm_summary

- **Status:** pass
- **Duration:** 9507ms
- **Tokens:** 2509
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['get_crm_summary'], actual=['get_crm_summary']
- **Tools called:** expected=['get_crm_summary'], actual=['get_crm_summary']
- **Response content:** score=1/1, missing=[]
- **Response:** Here's the CRM overview:  ### Customers - **Total Customers:** 156   - **Active:** 142   - **Leads:** 8   - **Prospects:** 4   - **Inactive:** 2  ### Deals - **Open Pipeline Value:** $127,500.00   - *...

### task_creation

- **Status:** pass
- **Duration:** 8814ms
- **Tokens:** 2844
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['create_task'], actual=['create_task']
- **Tools called:** expected=['create_task'], actual=['create_task']
- **Response content:** score=1/1, missing=[]
- **Response:** The follow-up task to call Joao Santos has been successfully created. Here are the details:  - **Title:** Follow-up call with Joao Santos - **Due Date:** 2026-02-14 - **Priority:** High - **Task ID:**...

### calendar_booking

- **Status:** partial
- **Duration:** 8078ms
- **Tokens:** 2681
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['query_calendar', 'create_calendar_event'], actual=['query_calendar']
- **Tools called:** expected=['query_calendar', 'create_calendar_event'], actual=['query_calendar']
- **Response content:** score=2/2, missing=[]
- **Response:** On February 10, 2026, the following appointments are scheduled:  1. **Haircut** - Maria Silva | 09:00 - 10:00 2. **Beard Trim** - Joao Santos | 10:30 - 11:00 3. **Hair Color** - Ana Oliveira | 14:00 -...

### customer_lookup

- **Status:** pass
- **Duration:** 6588ms
- **Tokens:** 3071
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Wasted loads:** ['get_customer_context']
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['ANY(query_customers | get_customer_context)'], actual=['get_customer_context', 'query_customers']
- **Tools called:** expected=['ANY(query_customers | get_customer_context)'], actual=['query_customers']
- **Response content:** score=n/a, missing=[]
- **Response:** The customer with the phone number **+5511999998888** is **Maria Silva**, and their status is **Active**....

### deal_creation

- **Status:** pass
- **Duration:** 6418ms
- **Tokens:** 2485
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['create_deal'], actual=['create_deal']
- **Tools called:** expected=['create_deal'], actual=['create_deal']
- **Response content:** score=1/1, missing=[]
- **Response:** The deal "Enterprise Plan for Construtora ABC" has been successfully created with a value of R$25,000 at the Proposal stage. The deal ID is **2f182be0-0f66-47a0-9a06-48465adf070f**....

### cross_category

- **Status:** pass
- **Duration:** 15047ms
- **Tokens:** 5273
- **Tool calls:** 6 (2 meta + 4 business)
- **Meta overhead:** 33% | **Efficiency:** 67%
- **Hit ceiling:** False
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['query_calendar', 'create_task'], actual=['create_task', 'query_calendar']
- **Tools called:** expected=['query_calendar', 'create_task'], actual=['query_calendar', 'create_task']
- **Response content:** score=n/a, missing=[]
- **Response:** I checked the calendar for February 6, 2026, and found three appointments. Follow-up tasks have been created for each customer:  1. **Follow-up with Maria Silva**    - **Due Date:** February 13, 2026 ...

### multi_tool_load

- **Status:** fail
- **Duration:** 12022ms
- **Tokens:** 5155
- **Tool calls:** 5 (2 meta + 3 business)
- **Meta overhead:** 40% | **Efficiency:** 60%
- **Hit ceiling:** False
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['load_tools']
- **Tools loaded:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_deal', 'create_task']
- **Tools called:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_deal', 'create_task']
- **Response content:** score=n/a, missing=[]
- **Response:** Here's what I've accomplished:  1. **Customer Registration**: A task has been created to register Ana Oliveira as a new customer.    - **Task ID**: d15ec720-1131-459b-9eee-83d7c187cb34    - **Due Date...

### category_browse

- **Status:** pass
- **Duration:** 6706ms
- **Tokens:** 3644
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Wasted loads:** ['query_calendar']
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['create_calendar_event'], actual=['create_calendar_event', 'query_calendar']
- **Tools called:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Response content:** score=n/a, missing=[]
- **Response:** The event "Team Meeting" has been successfully created for tomorrow at 10 AM, lasting for 1 hour. If you need any further assistance, feel free to ask!...

### group_load

- **Status:** pass
- **Duration:** 7994ms
- **Tokens:** 4051
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Wasted loads:** ['create_calendar_event', 'delete_calendar_event', 'update_calendar_event']
- **Meta-calls:** expected=['find_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['ANY(query_calendar)'], actual=['create_calendar_event', 'delete_calendar_event', 'query_calendar', 'update_calendar_event']
- **Tools called:** expected=['ANY(query_calendar)'], actual=['query_calendar']
- **Response content:** score=n/a, missing=[]
- **Response:** On **February 10, 2026**, your schedule includes the following events:  1. **Haircut** with Maria Silva from **09:00 to 10:00** 2. **Beard Trim** with Joao Santos from **10:30 to 11:00** 3. **Hair Col...

### customer_update

- **Status:** pass
- **Duration:** 6644ms
- **Tokens:** 2637
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['update_customer'], actual=['update_customer']
- **Tools called:** expected=['update_customer'], actual=['update_customer']
- **Response content:** score=1/1, missing=[]
- **Response:** Maria Silva's email address has been successfully updated to **maria.silva@newco.com**. If you need any further assistance, feel free to ask!...

### deal_lifecycle

- **Status:** partial
- **Duration:** 53431ms
- **Tokens:** 48516
- **Tool calls:** 40 (2 meta + 38 business)
- **Meta overhead:** 5% | **Efficiency:** 95%
- **Hit ceiling:** False
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['query_deals', 'ANY(update_deal | delete_deal)'], actual=['query_deals']
- **Tools called:** expected=['query_deals', 'ANY(update_deal | delete_deal)'], actual=['query_deals']
- **Response content:** score=n/a, missing=[]
- **Response:** [Tool executions completed without a final assistant message. Review returned payloads for actionable results.]...

### task_cleanup

- **Status:** pass
- **Duration:** 13052ms
- **Tokens:** 4765
- **Tool calls:** 7 (4 meta + 3 business)
- **Meta overhead:** 57% | **Efficiency:** 43%
- **Hit ceiling:** False
- **Redundant browses:** 2
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['query_tasks', 'ANY(delete_task | update_task)'], actual=['delete_task', 'query_tasks', 'update_task']
- **Tools called:** expected=['query_tasks', 'ANY(delete_task | update_task)'], actual=['query_tasks', 'delete_task', 'update_task']
- **Response content:** score=n/a, missing=[]
- **Response:** Here's the summary of the actions taken:  1. **Checked Overdue Tasks**: There are 12 overdue tasks, including:    - **Call back client re: invoice** (Due: 2026-01-28, Priority: High)    - **Send contr...

### session_persistence

- **Status:** pass
- **Duration:** 8743ms
- **Tokens:** 2710
- **Tool calls:** 4 (2 meta + 2 business)
- **Meta overhead:** 50% | **Efficiency:** 50%
- **Hit ceiling:** False
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['get_weather'], actual=['get_weather']
- **Tools called:** expected=['get_weather'], actual=['get_weather']
- **Response content:** score=1/1, missing=[]
- **Response:** The weather in Tokyo is currently sunny, with a temperature of 22°C....
