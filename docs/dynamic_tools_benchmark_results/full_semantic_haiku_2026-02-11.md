# Dynamic Tool Calling Benchmark Report

**Model:** `anthropic/claude-haiku-4-5-20251001`
**Date:** 2026-02-11 19:21:37 UTC
**Cases:** 13

## Summary

| Metric | Value |
|--------|-------|
| Pass | 12 |
| Partial | 1 |
| Fail | 0 |
| Error | 0 |
| Total time | 236975ms |
| Total tokens | 100504 |

## Results

| Status | Case | Protocol | Loading | Usage | Response | Overall | Calls | Meta% | Eff% | Ceiling | Time | Tokens |
|--------|------|----------|---------|-------|----------|---------|-------|-------|------|---------|------|--------|
| PASS | crm_summary | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 3 | 67% | 33% | - | 9228ms | 5734 |
| PASS | task_creation | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 4 | 75% | 25% | - | 8121ms | 7539 |
| PART | calendar_booking | 2/2 | 2/2 | 1/2 | 2/2 | 7/8 | 3 | 67% | 33% | - | 6580ms | 6146 |
| PASS | customer_lookup | 2/2 | 1/1 | 1/1 | n/a | 4/4 | 4 | 75% | 25% | - | 31309ms | 7148 |
| PASS | deal_creation | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 4 | 75% | 25% | - | 8574ms | 7267 |
| PASS | cross_category | 2/2 | 2/2 | 2/2 | n/a | 6/6 | 9 | 56% | 44% | - | 69375ms | 12891 |
| PASS | multi_tool_load | 2/2 | 3/3 | 3/3 | n/a | 8/8 | 7 | 57% | 43% | - | 48247ms | 10716 |
| PASS | category_browse | 2/2 | 1/1 | 1/1 | n/a | 4/4 | 4 | 75% | 25% | - | 7605ms | 7502 |
| PASS | group_load | 1/1 | 1/1 | 1/1 | n/a | 3/3 | 3 | 67% | 33% | - | 6008ms | 6734 |
| PASS | customer_update | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 4 | 75% | 25% | - | 8144ms | 7196 |
| PASS | deal_lifecycle | 2/2 | 2/2 | 2/2 | n/a | 6/6 | 5 | 40% | 60% | - | 7963ms | 6800 |
| PASS | task_cleanup | 2/2 | 2/2 | 2/2 | n/a | 6/6 | 7 | 57% | 43% | - | 8957ms | 7999 |
| PASS | session_persistence | 2/2 | 1/1 | 1/1 | 1/1 | 5/5 | 4 | 50% | 50% | - | 16864ms | 6832 |

## Failure Details

### calendar_booking [PARTIAL]

- **Tags:** multi-tool
- **Missing called:** ['create_calendar_event']
- **Response preview:** I can see your calendar for February 10, 2026. There are already 3 appointments scheduled that day: - 9:00-10:00: Haircut - Maria Silva - 10:30-11:00: Beard Trim - Joao Santos - 14:00-15:30: Hair Color - Ana Oliveira  The 2pm (14:00) slot is currently occupied by Ana Oliveira's Hair Color appointmen

## Efficiency Analysis

| Case | Calls | Meta | Business | Overhead% | Efficiency% | Redundant Browses | Wasted Loads | Ceiling |
|------|-------|------|----------|-----------|-------------|-------------------|--------------|---------|
| crm_summary | 3 | 2 | 1 | 67% | 33% | 0 | query_customers | - |
| task_creation | 4 | 3 | 1 | 75% | 25% | 1 | - | - |
| calendar_booking | 3 | 2 | 1 | 67% | 33% | 0 | create_calendar_event | - |
| customer_lookup | 4 | 3 | 1 | 75% | 25% | 1 | - | - |
| deal_creation | 4 | 3 | 1 | 75% | 25% | 1 | - | - |
| cross_category | 9 | 5 | 4 | 56% | 44% | 3 | - | - |
| multi_tool_load | 7 | 4 | 3 | 57% | 43% | 2 | - | - |
| category_browse | 4 | 3 | 1 | 75% | 25% | 1 | - | - |
| group_load | 3 | 2 | 1 | 67% | 33% | 0 | create_calendar_event, delete_calendar_event, update_calendar_event | - |
| customer_update | 4 | 3 | 1 | 75% | 25% | 1 | - | - |
| deal_lifecycle | 5 | 2 | 3 | 40% | 60% | 0 | - | - |
| task_cleanup | 7 | 4 | 3 | 57% | 43% | 2 | - | - |
| session_persistence | 4 | 2 | 2 | 50% | 50% | 0 | - | - |

**Aggregate:** 61 calls (38 meta + 23 business) | Ceiling hits: 0 | Redundant browses: 12 | Wasted loads: 5

## Per-Case Details

### crm_summary

- **Status:** pass
- **Duration:** 9228ms
- **Tokens:** 5734
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Wasted loads:** ['query_customers']
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['get_crm_summary'], actual=['get_crm_summary', 'query_customers']
- **Tools called:** expected=['get_crm_summary'], actual=['get_crm_summary']
- **Response content:** score=1/1, missing=[]
- **Response:** ## CRM Overview  **You have 156 total customers** in your CRM. Here's the breakdown:  ### Customer Status - **Active:** 142 customers - **Leads:** 8 - **Prospects:** 4 - **Inactive:** 2  ### Deal Pipe...

### task_creation

- **Status:** pass
- **Duration:** 8121ms
- **Tokens:** 7539
- **Tool calls:** 4 (3 meta + 1 business)
- **Meta overhead:** 75% | **Efficiency:** 25%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['create_task'], actual=['create_task']
- **Tools called:** expected=['create_task'], actual=['create_task']
- **Response content:** score=1/1, missing=[]
- **Response:** Perfect! I've successfully created a follow-up task with the following details:  - **Title:** Call Joao Santos - **Due Date:** February 14, 2026 - **Priority:** High - **Task ID:** 4917eac9-8859-4e2c-...

### calendar_booking

- **Status:** partial
- **Duration:** 6580ms
- **Tokens:** 6146
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Wasted loads:** ['create_calendar_event']
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['query_calendar', 'create_calendar_event'], actual=['create_calendar_event', 'query_calendar']
- **Tools called:** expected=['query_calendar', 'create_calendar_event'], actual=['query_calendar']
- **Response content:** score=2/2, missing=[]
- **Response:** I can see your calendar for February 10, 2026. There are already 3 appointments scheduled that day: - 9:00-10:00: Haircut - Maria Silva - 10:30-11:00: Beard Trim - Joao Santos - 14:00-15:30: Hair Colo...

### customer_lookup

- **Status:** pass
- **Duration:** 31309ms
- **Tokens:** 7148
- **Tool calls:** 4 (3 meta + 1 business)
- **Meta overhead:** 75% | **Efficiency:** 25%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['ANY(query_customers | get_customer_context)'], actual=['query_customers']
- **Tools called:** expected=['ANY(query_customers | get_customer_context)'], actual=['query_customers']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! I found the customer. Here are the details:  **Name:** Maria Silva   **Status:** Active  The customer with phone number +5511999998888 is Maria Silva, and their current status in the CRM is A...

### deal_creation

- **Status:** pass
- **Duration:** 8574ms
- **Tokens:** 7267
- **Tool calls:** 4 (3 meta + 1 business)
- **Meta overhead:** 75% | **Efficiency:** 25%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['create_deal'], actual=['create_deal']
- **Tools called:** expected=['create_deal'], actual=['create_deal']
- **Response content:** score=1/1, missing=[]
- **Response:** Perfect! ✅ I've successfully registered the new deal with the following details:  - **Deal Name:** Enterprise Plan for Construtora ABC - **Amount:** R$25,000.00 - **Stage:** Proposal - **Deal ID:** 2f...

### cross_category

- **Status:** pass
- **Duration:** 69375ms
- **Tokens:** 12891
- **Tool calls:** 9 (5 meta + 4 business)
- **Meta overhead:** 56% | **Efficiency:** 44%
- **Hit ceiling:** False
- **Redundant browses:** 3
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['query_calendar', 'create_task'], actual=['create_task', 'query_calendar']
- **Tools called:** expected=['query_calendar', 'create_task'], actual=['query_calendar', 'create_task']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! I've completed your request. Here's a summary:  **Calendar Check for 2026-02-06:** - 3 appointments found  **Follow-up Tasks Created:** 1. ✅ **Follow-up: Haircut - Maria Silva** (Due: 2026-02...

### multi_tool_load

- **Status:** pass
- **Duration:** 48247ms
- **Tokens:** 10716
- **Tool calls:** 7 (4 meta + 3 business)
- **Meta overhead:** 57% | **Efficiency:** 43%
- **Hit ceiling:** False
- **Redundant browses:** 2
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_customer', 'create_deal', 'create_task']
- **Tools called:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_customer', 'create_deal', 'create_task']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! ✅ All three items have been successfully created:  1. **Customer**: Ana Oliveira (ana@example.com)    - ID: 878e680e-ca2b-4392-bb32-fb3089bd53e2  2. **Deal**: Premium Package - $5,000.00    -...

### category_browse

- **Status:** pass
- **Duration:** 7605ms
- **Tokens:** 7502
- **Tool calls:** 4 (3 meta + 1 business)
- **Meta overhead:** 75% | **Efficiency:** 25%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Tools called:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! ✅ I've successfully created your calendar event:  **Team Meeting** - **Date & Time:** Tomorrow (January 10, 2025) at 10:00 AM - **Duration:** 1 hour (ends at 11:00 AM) - **Event ID:** 16dd307...

### group_load

- **Status:** pass
- **Duration:** 6008ms
- **Tokens:** 6734
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Wasted loads:** ['create_calendar_event', 'delete_calendar_event', 'update_calendar_event']
- **Meta-calls:** expected=['find_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['ANY(query_calendar)'], actual=['create_calendar_event', 'delete_calendar_event', 'query_calendar', 'update_calendar_event']
- **Tools called:** expected=['ANY(query_calendar)'], actual=['query_calendar']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! Here's your schedule for **February 10, 2026**:  1. **Haircut** - Maria Silva    - Time: 9:00 AM - 10:00 AM  2. **Beard Trim** - Joao Santos    - Time: 10:30 AM - 11:00 AM  3. **Hair Color** ...

### customer_update

- **Status:** pass
- **Duration:** 8144ms
- **Tokens:** 7196
- **Tool calls:** 4 (3 meta + 1 business)
- **Meta overhead:** 75% | **Efficiency:** 25%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['update_customer'], actual=['update_customer']
- **Tools called:** expected=['update_customer'], actual=['update_customer']
- **Response content:** score=1/1, missing=[]
- **Response:** Perfect! I've successfully updated Maria Silva's email address to maria.silva@newco.com. Her customer record (ID: c1-mock-uuid) has been updated in the CRM system....

### deal_lifecycle

- **Status:** pass
- **Duration:** 7963ms
- **Tokens:** 6800
- **Tool calls:** 5 (2 meta + 3 business)
- **Meta overhead:** 40% | **Efficiency:** 60%
- **Hit ceiling:** False
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['query_deals', 'ANY(update_deal | delete_deal)'], actual=['delete_deal', 'query_deals', 'update_deal']
- **Tools called:** expected=['query_deals', 'ANY(update_deal | delete_deal)'], actual=['query_deals', 'update_deal', 'delete_deal']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! I've completed all the tasks:  **Current Deals:** - **Premium Package** (d1-mock) - Maria Silva | $5,000 | Stage: Proposal - **Enterprise Plan** (d2-mock) - Joao Santos | $12,500 | Stage: Neg...

### task_cleanup

- **Status:** pass
- **Duration:** 8957ms
- **Tokens:** 7999
- **Tool calls:** 7 (4 meta + 3 business)
- **Meta overhead:** 57% | **Efficiency:** 43%
- **Hit ceiling:** False
- **Redundant browses:** 2
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['query_tasks', 'ANY(delete_task | update_task)'], actual=['delete_task', 'query_tasks', 'update_task']
- **Tools called:** expected=['query_tasks', 'ANY(delete_task | update_task)'], actual=['query_tasks', 'delete_task', 'update_task']
- **Response content:** score=n/a, missing=[]
- **Response:** Perfect! I've completed all the requested actions:  **Summary:**  ✅ **Overdue Tasks Checked:** You have 12 overdue tasks out of 45 open tasks. The two most critical ones were:    - "Call back client r...

### session_persistence

- **Status:** pass
- **Duration:** 16864ms
- **Tokens:** 6832
- **Tool calls:** 4 (2 meta + 2 business)
- **Meta overhead:** 50% | **Efficiency:** 50%
- **Hit ceiling:** False
- **Meta-calls:** expected=['find_tools', 'load_tools'], actual=['find_tools', 'load_tools']
- **Tools loaded:** expected=['get_weather'], actual=['get_weather']
- **Tools called:** expected=['get_weather'], actual=['get_weather']
- **Response content:** score=1/1, missing=[]
- **Response:** The weather in Tokyo is currently **sunny** with a temperature of **22°C** (approximately 72°F)....
