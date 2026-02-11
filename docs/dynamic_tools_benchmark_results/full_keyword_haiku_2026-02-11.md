# Dynamic Tool Calling Benchmark Report

**Model:** `anthropic/claude-haiku-4-5-20251001`
**Date:** 2026-02-11 19:20:45 UTC
**Cases:** 13

## Summary

| Metric | Value |
|--------|-------|
| Pass | 13 |
| Partial | 0 |
| Fail | 0 |
| Error | 0 |
| Total time | 186490ms |
| Total tokens | 132314 |

## Results

| Status | Case | Protocol | Loading | Usage | Response | Overall | Calls | Meta% | Eff% | Ceiling | Time | Tokens |
|--------|------|----------|---------|-------|----------|---------|-------|-------|------|---------|------|--------|
| PASS | crm_summary | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 4 | 75% | 25% | - | 6328ms | 11393 |
| PASS | task_creation | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 3 | 67% | 33% | - | 4740ms | 6823 |
| PASS | calendar_booking | 2/2 | 2/2 | 2/2 | 2/2 | 8/8 | 4 | 50% | 50% | - | 7174ms | 11525 |
| PASS | customer_lookup | 2/2 | 1/1 | 1/1 | n/a | 4/4 | 3 | 67% | 33% | - | 4306ms | 6708 |
| PASS | deal_creation | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 3 | 67% | 33% | - | 5665ms | 8070 |
| PASS | cross_category | 2/2 | 2/2 | 2/2 | n/a | 6/6 | 7 | 43% | 57% | - | 30984ms | 15608 |
| PASS | multi_tool_load | 2/2 | 3/3 | 3/3 | n/a | 8/8 | 7 | 57% | 43% | - | 62193ms | 19246 |
| PASS | category_browse | 2/2 | 1/1 | 1/1 | n/a | 4/4 | 3 | 67% | 33% | - | 6155ms | 7697 |
| PASS | group_load | 1/1 | 1/1 | 1/1 | n/a | 3/3 | 3 | 67% | 33% | - | 4634ms | 8957 |
| PASS | customer_update | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 3 | 67% | 33% | - | 3710ms | 6889 |
| PASS | deal_lifecycle | 2/2 | 2/2 | 2/2 | n/a | 6/6 | 5 | 40% | 60% | - | 6405ms | 9301 |
| PASS | task_cleanup | 2/2 | 2/2 | 2/2 | n/a | 6/6 | 5 | 40% | 60% | - | 38226ms | 9494 |
| PASS | session_persistence | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 4 | 50% | 50% | - | 5970ms | 10603 |

## Efficiency Analysis

| Case | Calls | Meta | Business | Overhead% | Efficiency% | Redundant Browses | Wasted Loads | Ceiling |
|------|-------|------|----------|-----------|-------------|-------------------|--------------|---------|
| crm_summary | 4 | 3 | 1 | 75% | 25% | 1 | - | - |
| task_creation | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| calendar_booking | 4 | 2 | 2 | 50% | 50% | 0 | - | - |
| customer_lookup | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| deal_creation | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| cross_category | 7 | 3 | 4 | 43% | 57% | 1 | - | - |
| multi_tool_load | 7 | 4 | 3 | 57% | 43% | 2 | - | - |
| category_browse | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| group_load | 3 | 2 | 1 | 67% | 33% | 0 | create_calendar_event, delete_calendar_event, update_calendar_event | - |
| customer_update | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| deal_lifecycle | 5 | 2 | 3 | 40% | 60% | 0 | - | - |
| task_cleanup | 5 | 2 | 3 | 40% | 60% | 0 | - | - |
| session_persistence | 4 | 2 | 2 | 50% | 50% | 0 | - | - |

**Aggregate:** 54 calls (30 meta + 24 business) | Ceiling hits: 0 | Redundant browses: 4 | Wasted loads: 3

## Per-Case Details

### crm_summary

- **Status:** pass
- **Duration:** 6328ms
- **Tokens:** 11393
- **Tool calls:** 4 (3 meta + 1 business)
- **Meta overhead:** 75% | **Efficiency:** 25%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['get_crm_summary'], actual=['get_crm_summary']
- **Tools called:** expected=['get_crm_summary'], actual=['get_crm_summary']
- **Response content:** score=1/1, missing=[]
- **Response:** ## CRM Overview  Great! Here's your complete CRM summary:  ### **Customers: 156 Total** - **Active:** 142 - **Leads:** 8 - **Prospects:** 4 - **Inactive:** 2  ### **Deal Pipeline: $127,500.00 Open** -...

### task_creation

- **Status:** pass
- **Duration:** 4740ms
- **Tokens:** 6823
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_task'], actual=['create_task']
- **Tools called:** expected=['create_task'], actual=['create_task']
- **Response content:** score=1/1, missing=[]
- **Response:** Perfect! I've successfully created the follow-up task with the following details:  - **Title:** Call Joao Santos - **Due Date:** February 14, 2026 - **Priority:** High - **Task ID:** a4099bc2-4f4a-44d...

### calendar_booking

- **Status:** pass
- **Duration:** 7174ms
- **Tokens:** 11525
- **Tool calls:** 4 (2 meta + 2 business)
- **Meta overhead:** 50% | **Efficiency:** 50%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_calendar', 'create_calendar_event'], actual=['create_calendar_event', 'query_calendar']
- **Tools called:** expected=['query_calendar', 'create_calendar_event'], actual=['query_calendar', 'create_calendar_event']
- **Response content:** score=2/2, missing=[]
- **Response:** Perfect! ✅ I've successfully booked the haircut appointment for Maria Silva on February 10, 2026 at 2:00 PM.   **Appointment Details:** - **Customer:** Maria Silva - **Service:** Haircut - **Date:** F...

### customer_lookup

- **Status:** pass
- **Duration:** 4306ms
- **Tokens:** 6708
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['ANY(query_customers | get_customer_context)'], actual=['query_customers']
- **Tools called:** expected=['ANY(query_customers | get_customer_context)'], actual=['query_customers']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! I found the customer with phone number +5511999998888.  **Customer Details:** - **Name:** Maria Silva - **Status:** Active...

### deal_creation

- **Status:** pass
- **Duration:** 5665ms
- **Tokens:** 8070
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_deal'], actual=['create_deal']
- **Tools called:** expected=['create_deal'], actual=['create_deal']
- **Response content:** score=1/1, missing=[]
- **Response:** Perfect! ✅ I've successfully registered the new deal with the following details:  - **Deal Name:** Enterprise Plan for Construtora ABC - **Amount:** R$25,000.00 - **Stage:** Proposal - **Deal ID:** db...

### cross_category

- **Status:** pass
- **Duration:** 30984ms
- **Tokens:** 15608
- **Tool calls:** 7 (3 meta + 4 business)
- **Meta overhead:** 43% | **Efficiency:** 57%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_calendar', 'create_task'], actual=['create_task', 'query_calendar']
- **Tools called:** expected=['query_calendar', 'create_task'], actual=['query_calendar', 'create_task']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! I've completed your request. Here's a summary:  **Calendar Check for 2026-02-06:** Found 3 appointments: 1. **Maria Silva** - Haircut (09:00 - 10:00) 2. **Joao Santos** - Beard Trim (10:30 - ...

### multi_tool_load

- **Status:** pass
- **Duration:** 62193ms
- **Tokens:** 19246
- **Tool calls:** 7 (4 meta + 3 business)
- **Meta overhead:** 57% | **Efficiency:** 43%
- **Hit ceiling:** False
- **Redundant browses:** 2
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_customer', 'create_deal', 'create_task']
- **Tools called:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_customer', 'create_deal', 'create_task']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! ✅ All three items have been successfully created:  1. **Customer**: Ana Oliveira (ana@example.com)    - ID: 85592b33-eb83-4d84-af76-86c85eb8902c  2. **Deal**: Premium Package - $5,000.00    -...

### category_browse

- **Status:** pass
- **Duration:** 6155ms
- **Tokens:** 7697
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Tools called:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! ✅ I've successfully created your calendar event:  **Team Meeting** - **Date & Time:** Tomorrow (January 10, 2025) at 10:00 AM - **Duration:** 1 hour (ends at 11:00 AM) - **Event ID:** eb01131...

### group_load

- **Status:** pass
- **Duration:** 4634ms
- **Tokens:** 8957
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Wasted loads:** ['create_calendar_event', 'delete_calendar_event', 'update_calendar_event']
- **Meta-calls:** expected=['browse_toolkit'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['ANY(query_calendar)'], actual=['create_calendar_event', 'delete_calendar_event', 'query_calendar', 'update_calendar_event']
- **Tools called:** expected=['ANY(query_calendar)'], actual=['query_calendar']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! Here's your schedule for **February 10, 2026**:  1. **Haircut** - Maria Silva      9:00 AM - 10:00 AM  2. **Beard Trim** - Joao Santos      10:30 AM - 11:00 AM  3. **Hair Color** - Ana Olivei...

### customer_update

- **Status:** pass
- **Duration:** 3710ms
- **Tokens:** 6889
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['update_customer'], actual=['update_customer']
- **Tools called:** expected=['update_customer'], actual=['update_customer']
- **Response content:** score=1/1, missing=[]
- **Response:** Perfect! I've successfully updated Maria Silva's email address to **maria.silva@newco.com**. The customer record with ID `c1-mock-uuid` has been updated....

### deal_lifecycle

- **Status:** pass
- **Duration:** 6405ms
- **Tokens:** 9301
- **Tool calls:** 5 (2 meta + 3 business)
- **Meta overhead:** 40% | **Efficiency:** 60%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_deals', 'ANY(update_deal | delete_deal)'], actual=['delete_deal', 'query_deals', 'update_deal']
- **Tools called:** expected=['query_deals', 'ANY(update_deal | delete_deal)'], actual=['query_deals', 'update_deal', 'delete_deal']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! I've completed all the tasks:  **Deals Summary:** Your pipeline currently has 3 deals totaling $19,000: 1. **Premium Package** (d1-mock) - Maria Silva | $5,000 | Stage: Proposal 2. **Enterpri...

### task_cleanup

- **Status:** pass
- **Duration:** 38226ms
- **Tokens:** 9494
- **Tool calls:** 5 (2 meta + 3 business)
- **Meta overhead:** 40% | **Efficiency:** 60%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_tasks', 'ANY(delete_task | update_task)'], actual=['delete_task', 'query_tasks', 'update_task']
- **Tools called:** expected=['query_tasks', 'ANY(delete_task | update_task)'], actual=['query_tasks', 'delete_task', 'update_task']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! I've completed all the requested actions:  **Overdue Tasks Summary:** - You have **12 overdue tasks** out of 45 open tasks - The two most urgent overdue tasks were:   1. "Call back client re:...

### session_persistence

- **Status:** pass
- **Duration:** 5970ms
- **Tokens:** 10603
- **Tool calls:** 4 (2 meta + 2 business)
- **Meta overhead:** 50% | **Efficiency:** 50%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['get_weather'], actual=['get_weather']
- **Tools called:** expected=['get_weather'], actual=['get_weather']
- **Response content:** score=1/1, missing=[]
- **Response:** The weather in Tokyo is currently **sunny** with a temperature of **22°C** (approximately 72°F)....
