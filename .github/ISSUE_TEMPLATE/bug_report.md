---
name: "Bug Report"
description: "Report a reproducible bug or unexpected behavior in llm-factory-toolkit"
title: "[BUG] <Brief description of bug>"
labels: ["bug"]
assignees: ''

---

**Describe the Bug**
A clear and concise description of what the bug is. What did you expect to happen? What actually happened?

**Steps to Reproduce**
Please provide a *minimal* reproducible example. This often involves:
1. Relevant Python code snippet demonstrating the issue.
2. Specific inputs (e.g., `messages` list, tool definitions).
3. The exact command or sequence of actions taken.

```python
# Your minimal Python code here
from llm_factory_toolkit import LLMClient

# Example setup...
client = LLMClient(provider_type='openai', model='gpt-4o-mini')
messages = [{"role": "user", "content": "..."}]

# The call that causes the bug
# result = await client.generate(messages)
```

**Expected Behavior**
A clear description of what you expected to happen.

**Actual Behavior**
What actually happened? Include any error messages or unexpected output.

**Error Output / Traceback**
```text
Paste the full traceback or error output here.
```

**Environment (please complete the following information):**
*   **llm-factory-toolkit Version:** [e.g., 0.1.0 - run `pip show llm-factory-toolkit`]
*   **Python Version:** [e.g., 3.10.4 - run `python --version`]
*   **Provider Used (if applicable):** [e.g., openai]
*   **Model Used (if applicable):** [e.g., gpt-4o-mini]
*   **Operating System:** [e.g., Windows 11, Ubuntu 22.04, macOS Sonoma]

**Additional Context**
Add any other context about the problem here. For example:
*   Did this work previously? If so, when did it stop working?
*   Are you using specific features like `response_format` or tool calls?
*   Any relevant dependencies and their versions (e.g., `openai`, `pydantic`).
