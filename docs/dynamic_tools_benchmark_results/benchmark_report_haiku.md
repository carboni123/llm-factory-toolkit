# Dynamic Tool Calling Benchmark Report

**Model:** `anthropic/claude-haiku-4-5-20251001`
**Date:** 2026-02-11 13:15:14 UTC
**Cases:** 10

## Summary

| Metric | Value |
|--------|-------|
| Pass | 9 |
| Partial | 1 |
| Fail | 0 |
| Error | 0 |
| Total time | 89873ms |
| Total tokens | 95271 |

## Results

| Status | Case | Protocol | Loading | Usage | Overall | Calls | Meta% | Eff% | Ceiling | Time | Tokens |
|--------|------|----------|---------|-------|---------|-------|-------|------|---------|------|--------|
| PASS | crm_summary | 2/2 | 1/1 | 1/1 | 4/4 | 4 | 75% | 25% | - | 10614ms | 8702 |
| PASS | task_creation | 2/2 | 1/1 | 1/1 | 4/4 | 3 | 67% | 33% | - | 7008ms | 6215 |
| PART | calendar_booking | 2/2 | 1/1 | 0/1 | 3/4 | 3 | 67% | 33% | - | 8488ms | 7822 |
| PASS | customer_lookup | 2/2 | 2/2 | 2/2 | 6/6 | 3 | 67% | 33% | - | 6996ms | 5810 |
| PASS | deal_creation | 2/2 | 1/1 | 1/1 | 4/4 | 3 | 67% | 33% | - | 5666ms | 6825 |
| PASS | cross_category | 2/2 | 2/2 | 2/2 | 6/6 | 9 | 56% | 44% | - | 14361ms | 19355 |
| PASS | multi_tool_load | 2/2 | 3/3 | 3/3 | 8/8 | 8 | 62% | 38% | - | 12029ms | 15806 |
| PASS | category_browse | 2/2 | 1/1 | 1/1 | 4/4 | 3 | 67% | 33% | - | 7392ms | 7101 |
| PASS | group_load | 1/1 | 1/1 | 1/1 | 3/3 | 3 | 67% | 33% | - | 6540ms | 8340 |
| PASS | session_persistence | 2/2 | 1/1 | 1/1 | 4/4 | 4 | 50% | 50% | - | 10779ms | 9295 |

## Failure Details

### calendar_booking [PARTIAL]

- **Tags:** multi-tool
- **Missing called:** ['create_calendar_event']
- **Response preview:** I can see your calendar for February 10, 2026. There are already 3 appointments scheduled that day: - 9:00-10:00: Haircut - Maria Silva - 10:30-11:00: Beard Trim - Joao Santos - 14:00-15:30: Hair Color - Ana Oliveira  The 2pm (14:00) slot is currently occupied by Ana Oliveira's Hair Color appointmen

## Efficiency Analysis

| Case | Calls | Meta | Business | Overhead% | Efficiency% | Redundant Browses | Wasted Loads | Ceiling |
|------|-------|------|----------|-----------|-------------|-------------------|--------------|---------|
| crm_summary | 4 | 3 | 1 | 75% | 25% | 1 | - | - |
| task_creation | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| calendar_booking | 3 | 2 | 1 | 67% | 33% | 0 | create_calendar_event | - |
| customer_lookup | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| deal_creation | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| cross_category | 9 | 5 | 4 | 56% | 44% | 3 | - | - |
| multi_tool_load | 8 | 5 | 3 | 62% | 38% | 3 | - | - |
| category_browse | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| group_load | 3 | 2 | 1 | 67% | 33% | 0 | create_calendar_event, delete_calendar_event, update_calendar_event | - |
| session_persistence | 4 | 2 | 2 | 50% | 50% | 0 | - | - |

**Aggregate:** 43 calls (27 meta + 16 business) | Ceiling hits: 0 | Redundant browses: 7 | Wasted loads: 4

## Per-Case Details

### crm_summary

- **Status:** pass
- **Duration:** 10614ms
- **Tokens:** 8702
- **Tool calls:** 4 (3 meta + 1 business)
- **Meta overhead:** 75% | **Efficiency:** 25%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['get_crm_summary'], actual=['get_crm_summary']
- **Tools called:** expected=['get_crm_summary'], actual=['get_crm_summary']
- **Response:** ## CRM Overview  Great! Here's your complete CRM overview:  ### **Customers: 156 Total** - **Active:** 142 - **Leads:** 8 - **Prospects:** 4 - **Inactive:** 2  ### **Deal Pipeline: $127,500.00 Open** ...

### task_creation

- **Status:** pass
- **Duration:** 7008ms
- **Tokens:** 6215
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_task'], actual=['create_task']
- **Tools called:** expected=['create_task'], actual=['create_task']
- **Response:** Perfect! I've successfully created the follow-up task with the following details:  - **Title:** Call Joao Santos - **Due Date:** February 14, 2026 - **Priority:** High - **Task ID:** 92719e7a-7598-4b8...

### calendar_booking

- **Status:** partial
- **Duration:** 8488ms
- **Tokens:** 7822
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Wasted loads:** ['create_calendar_event']
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_calendar_event'], actual=['create_calendar_event', 'query_calendar']
- **Tools called:** expected=['create_calendar_event'], actual=['query_calendar']
- **Response:** I can see your calendar for February 10, 2026. There are already 3 appointments scheduled that day: - 9:00-10:00: Haircut - Maria Silva - 10:30-11:00: Beard Trim - Joao Santos - 14:00-15:30: Hair Colo...

### customer_lookup

- **Status:** pass
- **Duration:** 6996ms
- **Tokens:** 5810
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_customers', 'get_customer_context'], actual=['query_customers']
- **Tools called:** expected=['query_customers', 'get_customer_context'], actual=['query_customers']
- **Response:** Perfect! I found the customer. Here are the details:  **Name:** Maria Silva   **Status:** Active  The customer with phone number +5511999998888 is Maria Silva, and their current status is Active....

### deal_creation

- **Status:** pass
- **Duration:** 5666ms
- **Tokens:** 6825
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_deal'], actual=['create_deal']
- **Tools called:** expected=['create_deal'], actual=['create_deal']
- **Response:** Perfect! ✅ I've successfully registered the new deal with the following details:  - **Deal Name:** Enterprise Plan for Construtora ABC - **Deal ID:** 088de91d-b5ce-4df3-984d-1360f5d6e985 - **Amount:**...

### cross_category

- **Status:** pass
- **Duration:** 14361ms
- **Tokens:** 19355
- **Tool calls:** 9 (5 meta + 4 business)
- **Meta overhead:** 56% | **Efficiency:** 44%
- **Hit ceiling:** False
- **Redundant browses:** 3
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_calendar', 'create_task'], actual=['create_task', 'query_calendar']
- **Tools called:** expected=['query_calendar', 'create_task'], actual=['query_calendar', 'create_task']
- **Response:** Perfect! I've completed your request. Here's a summary:  **Calendar Check for 2026-02-06:** Found 3 appointments: 1. **Maria Silva** - Haircut (09:00 - 10:00) 2. **Joao Santos** - Beard Trim (10:30 - ...

### multi_tool_load

- **Status:** pass
- **Duration:** 12029ms
- **Tokens:** 15806
- **Tool calls:** 8 (5 meta + 3 business)
- **Meta overhead:** 62% | **Efficiency:** 38%
- **Hit ceiling:** False
- **Redundant browses:** 3
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_customer', 'create_deal', 'create_task']
- **Tools called:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_customer', 'create_deal', 'create_task']
- **Response:** Perfect! I've successfully completed all three tasks:  ✅ **Customer Created:** - Name: Ana Oliveira - Email: ana@example.com - Customer ID: 21da1f29-5761-4c1f-b536-5239849fb5c5  ✅ **Deal Created:** - ...

### category_browse

- **Status:** pass
- **Duration:** 7392ms
- **Tokens:** 7101
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Tools called:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Response:** Perfect! ✅ I've successfully created your calendar event:  **Team Meeting** - **Date & Time:** Tomorrow (January 10, 2025) at 10:00 AM - **Duration:** 1 hour - **Event ID:** c77dec33-fbc0-430f-8d75-ad...

### group_load

- **Status:** pass
- **Duration:** 6540ms
- **Tokens:** 8340
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Wasted loads:** ['create_calendar_event', 'delete_calendar_event', 'update_calendar_event']
- **Meta-calls:** expected=['browse_toolkit'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_calendar'], actual=['create_calendar_event', 'delete_calendar_event', 'query_calendar', 'update_calendar_event']
- **Tools called:** expected=['query_calendar'], actual=['query_calendar']
- **Response:** Perfect! I've loaded all calendar-related tools and checked your schedule for February 10, 2026. Here's what you have scheduled:  **Schedule for 2026-02-10:**  1. **Haircut** - Maria Silva | 9:00 AM -...

### session_persistence

- **Status:** pass
- **Duration:** 10779ms
- **Tokens:** 9295
- **Tool calls:** 4 (2 meta + 2 business)
- **Meta overhead:** 50% | **Efficiency:** 50%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['get_weather'], actual=['get_weather']
- **Tools called:** expected=['get_weather'], actual=['get_weather']
- **Response:** The weather in Tokyo is currently **sunny** with a temperature of **22°C** (approximately 72°F)....
