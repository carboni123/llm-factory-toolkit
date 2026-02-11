# Dynamic Tool Calling Benchmark Report

**Model:** `openai/gpt-4o-mini`
**Date:** 2026-02-11 13:07:11 UTC
**Cases:** 10

## Summary

| Metric | Value |
|--------|-------|
| Pass | 6 |
| Partial | 1 |
| Fail | 3 |
| Error | 0 |
| Total time | 127012ms |
| Total tokens | 58050 |

## Results

| Status | Case | Protocol | Loading | Usage | Overall | Calls | Meta% | Eff% | Ceiling | Time | Tokens |
|--------|------|----------|---------|-------|---------|-------|-------|------|---------|------|--------|
| FAIL | crm_summary | 1/2 | 0/1 | 0/1 | 1/4 | 3 | 100% | 0% | - | 9757ms | 2752 |
| PASS | task_creation | 2/2 | 1/1 | 1/1 | 4/4 | 4 | 75% | 25% | - | 16041ms | 4662 |
| PASS | calendar_booking | 2/2 | 1/1 | 1/1 | 4/4 | 4 | 50% | 50% | - | 11183ms | 6305 |
| FAIL | customer_lookup | 1/2 | 0/2 | 0/2 | 1/6 | 2 | 100% | 0% | - | 6433ms | 1927 |
| PASS | deal_creation | 2/2 | 1/1 | 1/1 | 4/4 | 7 | 86% | 14% | - | 18380ms | 9837 |
| FAIL | cross_category | 1/2 | 0/2 | 0/2 | 1/6 | 3 | 100% | 0% | - | 10962ms | 2849 |
| PART | multi_tool_load | 2/2 | 2/3 | 2/3 | 6/8 | 11 | 82% | 18% | - | 27138ms | 15289 |
| PASS | category_browse | 2/2 | 1/1 | 1/1 | 4/4 | 3 | 67% | 33% | - | 7486ms | 4162 |
| PASS | group_load | 1/1 | 1/1 | 1/1 | 3/3 | 3 | 67% | 33% | - | 11410ms | 4992 |
| PASS | session_persistence | 2/2 | 1/1 | 1/1 | 4/4 | 4 | 50% | 50% | - | 8222ms | 5275 |

## Failure Details

### crm_summary [FAIL]

- **Tags:** smoke
- **Missing meta-calls:** ['load_tools']
- **Missing loaded:** ['get_crm_summary']
- **Missing called:** ['get_crm_summary']
- **Response preview:** It seems that I couldn't find any specific tools related to CRM or customer count in the current toolkit.   To assist you better, could you provide more details about the CRM system you are using? This might help in determining how to access the customer data you need.

### customer_lookup [FAIL]

- **Tags:** smoke
- **Missing meta-calls:** ['load_tools']
- **Missing loaded:** ['query_customers', 'get_customer_context']
- **Missing called:** ['query_customers', 'get_customer_context']
- **Response preview:** It seems that there are currently no available tools to search for customer contacts in the CRM system. If you have any other requests or need assistance with something else, feel free to let me know!

### cross_category [FAIL]

- **Tags:** cross-category
- **Missing meta-calls:** ['load_tools']
- **Missing loaded:** ['query_calendar', 'create_task']
- **Missing called:** ['query_calendar', 'create_task']
- **Response preview:** It seems that I couldn't find any tools related to checking calendars or managing tasks and appointments. If you have access to a specific calendar or task management system, please let me know how I can assist you further!

### multi_tool_load [PARTIAL]

- **Tags:** multi-tool
- **Missing loaded:** ['create_customer']
- **Missing called:** ['create_customer']
- **Response preview:** Here's what I've accomplished:  1. **Deal Created**:     - **Name**: Premium Package    - **Amount**: $5000    - **Deal ID**: 7f2e0cba-ff30-4d5f-a9d1-1079b68f0d5a  2. **Task Created**:     - **Title**: Send Proposal to Ana Oliveira    - **Due Date**: 2023-10-10    - **Priority**: High    - **Task ID

## Efficiency Analysis

| Case | Calls | Meta | Business | Overhead% | Efficiency% | Redundant Browses | Wasted Loads | Ceiling |
|------|-------|------|----------|-----------|-------------|-------------------|--------------|---------|
| crm_summary | 3 | 3 | 0 | 100% | 0% | 2 | - | - |
| task_creation | 4 | 3 | 1 | 75% | 25% | 1 | - | - |
| calendar_booking | 4 | 2 | 2 | 50% | 50% | 0 | - | - |
| customer_lookup | 2 | 2 | 0 | 100% | 0% | 1 | - | - |
| deal_creation | 7 | 6 | 1 | 86% | 14% | 4 | - | - |
| cross_category | 3 | 3 | 0 | 100% | 0% | 2 | - | - |
| multi_tool_load | 11 | 9 | 2 | 82% | 18% | 7 | - | - |
| category_browse | 3 | 2 | 1 | 67% | 33% | 0 | - | - |
| group_load | 3 | 2 | 1 | 67% | 33% | 0 | create_calendar_event, delete_calendar_event, update_calendar_event | - |
| session_persistence | 4 | 2 | 2 | 50% | 50% | 0 | - | - |

**Aggregate:** 44 calls (34 meta + 10 business) | Ceiling hits: 0 | Redundant browses: 17 | Wasted loads: 3

## Per-Case Details

### crm_summary

- **Status:** fail
- **Duration:** 9757ms
- **Tokens:** 2752
- **Tool calls:** 3 (3 meta + 0 business)
- **Meta overhead:** 100% | **Efficiency:** 0%
- **Hit ceiling:** False
- **Redundant browses:** 2
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit']
- **Tools loaded:** expected=['get_crm_summary'], actual=[]
- **Tools called:** expected=['get_crm_summary'], actual=[]
- **Response:** It seems that I couldn't find any specific tools related to CRM or customer count in the current toolkit.   To assist you better, could you provide more details about the CRM system you are using? Thi...

### task_creation

- **Status:** pass
- **Duration:** 16041ms
- **Tokens:** 4662
- **Tool calls:** 4 (3 meta + 1 business)
- **Meta overhead:** 75% | **Efficiency:** 25%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_task'], actual=['create_task']
- **Tools called:** expected=['create_task'], actual=['create_task']
- **Response:** The follow-up task to call Joao Santos has been successfully created. Here are the details:  - **Title:** Call Joao Santos - **Due Date:** 2026-02-14 - **Priority:** High  If you need anything else, f...

### calendar_booking

- **Status:** pass
- **Duration:** 11183ms
- **Tokens:** 6305
- **Tool calls:** 4 (2 meta + 2 business)
- **Meta overhead:** 50% | **Efficiency:** 50%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_calendar_event'], actual=['create_calendar_event', 'query_calendar']
- **Tools called:** expected=['create_calendar_event'], actual=['query_calendar', 'create_calendar_event']
- **Response:** The haircut appointment for Maria Silva has been successfully booked for **February 10, 2026, at 2:00 PM**. Here are the details:  - **Title:** Haircut Appointment for Maria Silva - **Start Time:** 2:...

### customer_lookup

- **Status:** fail
- **Duration:** 6433ms
- **Tokens:** 1927
- **Tool calls:** 2 (2 meta + 0 business)
- **Meta overhead:** 100% | **Efficiency:** 0%
- **Hit ceiling:** False
- **Redundant browses:** 1
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit']
- **Tools loaded:** expected=['query_customers', 'get_customer_context'], actual=[]
- **Tools called:** expected=['query_customers', 'get_customer_context'], actual=[]
- **Response:** It seems that there are currently no available tools to search for customer contacts in the CRM system. If you have any other requests or need assistance with something else, feel free to let me know!...

### deal_creation

- **Status:** pass
- **Duration:** 18380ms
- **Tokens:** 9837
- **Tool calls:** 7 (6 meta + 1 business)
- **Meta overhead:** 86% | **Efficiency:** 14%
- **Hit ceiling:** False
- **Redundant browses:** 4
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_deal'], actual=['create_deal']
- **Tools called:** expected=['create_deal'], actual=['create_deal']
- **Response:** The deal "Enterprise Plan for Construtora ABC" has been successfully created with a value of R$25,000 at the Proposal stage.   Deal ID: **5cab6181-5deb-4767-96d3-5c697779b4dc**....

### cross_category

- **Status:** fail
- **Duration:** 10962ms
- **Tokens:** 2849
- **Tool calls:** 3 (3 meta + 0 business)
- **Meta overhead:** 100% | **Efficiency:** 0%
- **Hit ceiling:** False
- **Redundant browses:** 2
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit']
- **Tools loaded:** expected=['query_calendar', 'create_task'], actual=[]
- **Tools called:** expected=['query_calendar', 'create_task'], actual=[]
- **Response:** It seems that I couldn't find any tools related to checking calendars or managing tasks and appointments. If you have access to a specific calendar or task management system, please let me know how I ...

### multi_tool_load

- **Status:** partial
- **Duration:** 27138ms
- **Tokens:** 15289
- **Tool calls:** 11 (9 meta + 2 business)
- **Meta overhead:** 82% | **Efficiency:** 18%
- **Hit ceiling:** False
- **Redundant browses:** 7
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_deal', 'create_task']
- **Tools called:** expected=['create_customer', 'create_deal', 'create_task'], actual=['create_deal', 'create_task']
- **Response:** Here's what I've accomplished:  1. **Deal Created**:     - **Name**: Premium Package    - **Amount**: $5000    - **Deal ID**: 7f2e0cba-ff30-4d5f-a9d1-1079b68f0d5a  2. **Task Created**:     - **Title**...

### category_browse

- **Status:** pass
- **Duration:** 7486ms
- **Tokens:** 4162
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Tools called:** expected=['create_calendar_event'], actual=['create_calendar_event']
- **Response:** The event "Team Meeting" has been successfully created for tomorrow at 10 AM, lasting for 1 hour. If you need any further assistance, feel free to ask!...

### group_load

- **Status:** pass
- **Duration:** 11410ms
- **Tokens:** 4992
- **Tool calls:** 3 (2 meta + 1 business)
- **Meta overhead:** 67% | **Efficiency:** 33%
- **Hit ceiling:** False
- **Wasted loads:** ['create_calendar_event', 'delete_calendar_event', 'update_calendar_event']
- **Meta-calls:** expected=['browse_toolkit'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['query_calendar'], actual=['create_calendar_event', 'delete_calendar_event', 'query_calendar', 'update_calendar_event']
- **Tools called:** expected=['query_calendar'], actual=['query_calendar']
- **Response:** On **February 10, 2026**, your schedule includes the following events:  1. **Haircut** with Maria Silva    - **Time:** 09:00 - 10:00  2. **Beard Trim** with Joao Santos    - **Time:** 10:30 - 11:00  3...

### session_persistence

- **Status:** pass
- **Duration:** 8222ms
- **Tokens:** 5275
- **Tool calls:** 4 (2 meta + 2 business)
- **Meta overhead:** 50% | **Efficiency:** 50%
- **Hit ceiling:** False
- **Meta-calls:** expected=['browse_toolkit', 'load_tools'], actual=['browse_toolkit', 'load_tools']
- **Tools loaded:** expected=['get_weather'], actual=['get_weather']
- **Tools called:** expected=['get_weather'], actual=['get_weather']
- **Response:** The weather in Tokyo is currently sunny, with a temperature of 22°C....
