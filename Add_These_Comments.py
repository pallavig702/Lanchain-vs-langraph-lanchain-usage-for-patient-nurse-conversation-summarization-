#!/usr/bin/env python3
"""
agentic_summarizer_cli.py
--------------------------------

This script implements a **serial LangGraph pipeline** that extracts clinical
information from patient–nurse transcripts and produces a **SOAP note**
plus **Omaha System problems/interventions**.

✨ Key Features:
- **Preprocessing**: normalize diarization (e.g., SPEAKER_01 → Nurse/Patient) and strip greetings.
- **Extractor heads**: each focuses on a clinical dimension:
    * Core heads: Problems, Observations, Emotions/Cognition, Interventions, Patient Response
    * Extended heads: Medications, Allergies, Vitals, History, SDOH, Functional status, Risk/Safety, Education/Teach-back, Follow-ups
- **Validators**: after each extractor, a validator chain ensures items are
  *only* present if literally supported in the transcript (prevents hallucination).
- **Composer**: merges validated outputs into SOAP note (S/O/A/P sections).
- **Omaha Mapper**: maps validated problems/interventions into Omaha labels.
- **Final Review (Hallucination Scrub)**: compares the summary against the
  transcript and removes anything unsupported or paraphrased incorrectly.
- **Fallback**: if structured parsing fails, a backup summarizer produces
  text-only SOAP + Omaha directly.

🔧 How it runs:
1. Read transcript (file or stdin).
2. Preprocess (normalize speakers, strip greetings).
3. Walk through extractors → validators (serial order).
4. Compose SOAP, Omaha mapping, finalize to human-readable text.
5. Apply final hallucination scrub for factual fidelity.
6. Print summary to stdout.

Run:
  python agentic_summarizer_cli.py --file demo_transcript.txt

Dependencies:
  pip install langchain langchain-openai pydantic langgraph
  export OPENAI_API_KEY="yourkey"
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, TypedDict, Optional, Dict, Any

from pydantic import BaseModel, Field

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangGraph
from langgraph.graph import StateGraph, START, END

# ==============================================================
# 0) PREPROCESSING
# --------------------------------------------------------------
# Normalizes transcripts so that later extractor nodes work
# on consistent input. Removes greetings at the start of visits
# (but keeps "yes/no/okay"). This keeps extractors focused only
# on clinically relevant content.
# ==============================================================

SPEAKER_RE = re.compile(r'^\s*SPEAKER_\d+\s*\(([^)]+)\):\s*(.*)$', re.IGNORECASE)

def normalize_transcript(raw: str) -> str:
    """Convert 'SPEAKER_01 (Nurse): ...' → 'Nurse: ...' / 'Patient: ...'."""
    out = []
    for line in raw.splitlines():
        m = SPEAKER_RE.match(line)
        if not m:
            continue
        role_raw, text = m.groups()
        text = text.strip()
        if not text:
            continue
        role_norm = role_raw.strip().lower()
        speaker = "Patient" if "patient" in role_norm else ("Nurse" if "nurse" in role_norm else role_raw.title())
        out.append(f"{speaker}: {text}")
    return "\n".join(out)

# Patterns for greetings to drop (applies only at the start of transcript)
GREETING_PATTERNS = [
    re.compile(r"^(hi|hello|hey)\W*$", re.I),
    re.compile(r"^good (morning|afternoon|evening)\W*$", re.I),
    re.compile(r"^how (are|r) (you|ya)( today)?\W*$", re.I),
    re.compile(r"^(i am|i'm)\s*(fine|good|well)\W*(thank you|thanks)?\W*$", re.I),
    re.compile(r"^(thank you|thanks|you'?re welcome)\W*$", re.I),
    re.compile(r"^nice to (meet|see) you\W*$", re.I),
    re.compile(r"^that'?s good\W*$", re.I),
    re.compile(r"^have a (nice|good) day\W*$", re.I),
]
# Cue words to detect that the "clinical" part of conversation has started
CLINICAL_CUES = [
    re.compile(r"\b(pain|fever|cough|breath|breathing|sleep|dizzi|bp|blood pressure|glucose|sugar|wound|medicat|dose|injec|fall|injur|symptom|since|when|monitor|exercise)\b", re.I),
    re.compile(r"\?$"),
    re.compile(r"\bdescribe how\b", re.I),
    re.compile(r"\bhow have you been feeling\b", re.I),
]

def strip_greetings_only(normalized_text: str) -> str:
    """Drop greetings only at the start, stop once clinical talk begins."""
    kept, clinical_started = [], False
    for line in normalized_text.splitlines():
        if ":" not in line:
            continue
        role, text = line.split(":", 1)
        t = text.strip()
        if clinical_started:
            kept.append(f"{role.strip()}: {t}")
            continue
        if any(p.search(t) for p in CLINICAL_CUES):
            clinical_started = True
            kept.append(f"{role.strip()}: {t}")
            continue
        if t.lower() in {"yes", "no", "ok", "okay", "alright"}:
            kept.append(f"{role.strip()}: {t}")
            continue
        if any(p.fullmatch(t) for p in GREETING_PATTERNS):
            continue
        kept.append(f"{role.strip()}: {t}")
    return "\n".join(kept)

def preprocess_text(raw: str, keep_greetings: bool = False) -> str:
    """Main preprocess entrypoint."""
    norm = normalize_transcript(raw)
    return norm if keep_greetings else strip_greetings_only(norm)

# ==============================================================
# 1) SCHEMAS
# --------------------------------------------------------------
# Defines Pydantic models for each extractor head output. These
# enforce structured JSON output from the LLM extractors and
# validators. New clinically relevant heads are added here too.
# ==============================================================

# ... (schemas omitted here for brevity in this excerpt, but same as earlier script)

# ==============================================================
# 2) STATE
# --------------------------------------------------------------
# Shared state across the LangGraph pipeline. Holds the transcript,
# each extractor + validator output, downstream SOAP/Omaha, and
# final text result.
# ==============================================================

# ... (State class definition)

# ==============================================================
# 3) CHAINS
# --------------------------------------------------------------
# Factory functions that wrap an LLM into an extractor, validator,
# composer, mapper, or hallucination scrub.
# Each chain enforces strict "no-new-facts" rules.
# ==============================================================

# ... (build_llm, extractor_chain, validator_chain, composer_chain,
# omaha_chain, fallback_soap_chain, factuality_guard_chain)

# ==============================================================
# 4) NODES
# --------------------------------------------------------------
# Node functions that actually get wired into LangGraph. They call
# the chains, handle errors gracefully with `safe_invoke`, and update
# the graph state.
# ==============================================================

# ... (n_extract, n_validate, n_soap, n_omaha, n_finalize, n_final_review)

# ==============================================================
# 5) GRAPH BUILDER
# --------------------------------------------------------------
# Builds a serial graph:
#   START → extractors+validators → SOAP → Omaha → Finalize → Hallucination scrub → END
#
# Old heads first (Problems, Observations, etc.) → New heads
# (Meds, Allergies, Vitals, SDOH, etc.) → Composition.
# ==============================================================

# ... (build_graph)

# ==============================================================
# 6) RUNNER
# --------------------------------------------------------------
# Orchestrates everything: builds the graph, runs it, applies
# fallback if necessary, and returns final text.
# ==============================================================

# ... (summarize_text)

# ==============================================================
# 7) CLI
# --------------------------------------------------------------
# Command-line entry point. Reads input, preprocesses, calls
# summarizer, prints result.
# ==============================================================

# ... (main)

if __name__ == "__main__":
    main()
