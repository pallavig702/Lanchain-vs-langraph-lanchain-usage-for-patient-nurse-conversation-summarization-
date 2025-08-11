## Agentic AI standardized summarization of patient-nurse conversation
#### ReAct versus CoT
**What we did:**  Each node is a single LLM call with a small CoT preamble (think silently), then passes structured output to the next node. LangGraph just orders the nodes.
**What we didn't do:** No action–observation loops, no tool calls, no web/EHR lookups inside the reasoning (i.e., no ReAct behavior).
**Why:** For transcript-only summarization, ReAct adds complexity without benefit. CoT + a fixed graph is simpler, cheaper, and more reproducible.
**When to consider ReAct later:** If you want the agent to query external tools (EHR, guidelines, drug DBs) mid-run, we can add a separate tool-using subgraph or a ReAct-style node. For now, it stays off.
