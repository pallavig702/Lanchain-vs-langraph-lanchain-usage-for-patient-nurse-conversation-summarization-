## Agentic AI standardized summarization of patient-nurse conversation
### ReAct versus CoT<br />
**What we did:**  Each node is a single LLM call with a small CoT preamble (think silently), then passes structured output to the next node. LangGraph just orders the nodes.<br />
**What we didn't do:** No action–observation loops, no tool calls, no web/EHR lookups inside the reasoning (i.e., no ReAct behavior).<br />
**Why:** For transcript-only summarization, ReAct adds complexity without benefit. CoT + a fixed graph is simpler, cheaper, and more reproducible.<br />
**When to consider ReAct later:** If you want the agent to query external tools (EHR, guidelines, drug DBs) mid-run, we can add a separate tool-using subgraph or a ReAct-style node. For now, it stays off.<br />

### Step 1: Preprocessing:

### Step 2: Langraph build on top of langchain: 
**Run** python Langchain_Langraph_Summarization.py -f InputProcessedTranscriptWithRoles.txt
* LangGraph = the orchestrator.
    * We use StateGraph, START, END to wire nodes into a graph (Problems → Observations → … → SOAP → Omaha → Finalize).
* LangChain = the building blocks inside each node.
    * We use ChatOpenAI, ChatPromptTemplate, and JsonOutputParser to make each extractor/composer chain.<br />
* In the script you can see it:<br />
  from langgraph.graph import StateGraph, START, END → graph wiring<br />
  from langchain_openai import ChatOpenAI + from langchain_core... → LLM calls, prompts, parsing<br />
* ** So: LangGraph sits on top of LangChain here—nodes are LangChain chains, and LangGraph coordinates how/when they run. You could swap a node for a plain Python function later; LangGraph doesn’t require LangChain, but they pair nicely for this pattern**.<br />
