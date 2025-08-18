#!/usr/bin/env python3
"""
Agentic Summarization Pipeline with Review Agent (Production-leaning, CLI-ready)

Inputs: raw transcript text (via --file or STDIN)
Outputs: SOAP note + Omaha mappings + warnings (stdout)

Key features
- Preprocess: speaker normalization + optional greeting stripping
- Extractor agents: problems, observations, emotion/cognition, interventions, patient response
- Composer agent: SOAP generator (with fallback)
- Ontology agent: Omaha mapper
- **Review agent**: structural coverage, semantic consistency, ontology alignment, bias/safety
- Finalizer: assembles human-readable output and appends warnings / QA checks

Requires:
  pip install langchain langchain-openai langchain-core langgraph pydantic
  export OPENAI_API_KEY=...

Note: This script uses JSON-only outputs via pydantic-bound parsers to avoid CoT leakage.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Optional

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph

# -----------------------------
# Preprocessing
# -----------------------------
SPEAKER_RE = re.compile(r"^(SPEAKER_\d+\s*\(([^)]+)\)|SPEAKER_\d+|Patient|Nurse)\s*:\s*(.*)$")

GREETING_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\b",
        r"\bthanks?\b",
        r"\bhow are you\b",
        r"\bnice to (meet|see) you\b",
    ]
]

CLINICAL_CUES = [
    re.compile(p, re.I)
    for p in [
        r"\bpain\b",
        r"\bmedication|dose|pill|insulin\b",
        r"\bblood pressure|bp|glucose|spo2|oxygen\b",
        r"\bshortness of breath|dyspnea|cough|wheeze\b",
        r"\bfall|dizzy|balance\b",
        r"\bassessment|plan|monitor\b",
    ]
]

def normalize_transcript(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        m = SPEAKER_RE.match(line.strip())
        if not m:
            if line.strip():
                lines.append(line.strip())
            continue
        speaker_full, role_hint, content = m.group(1), m.group(2), m.group(3)
        role = None
        if role_hint:
            role_l = role_hint.lower()
            if "nurse" in role_l:
                role = "Nurse"
            elif "patient" in role_l:
                role = "Patient"
        if not role:
            sf = speaker_full.lower()
            if "patient" in sf:
                role = "Patient"
            elif "nurse" in sf:
                role = "Nurse"
            else:
                # fallback based on id
                role = "Nurse" if "_01" in sf else "Patient"
        lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines)

def _looks_clinical(text: str) -> bool:
    return any(p.search(text) for p in CLINICAL_CUES)

def strip_greetings_only(normalized_text: str) -> str:
    out = []
    started = False
    for line in normalized_text.splitlines():
        if not started:
            if _looks_clinical(line):
                started = True
                out.append(line)
                continue
            # skip obvious greeting/pleasantry
            if any(p.search(line) for p in GREETING_PATTERNS):
                continue
            # allow short acknowledgements to pass through if not greeting
            if line.strip().lower() in {"ok", "okay", "yes", "no", "mm-hmm", "uh-huh"}:
                out.append(line)
            else:
                # keep neutral lines to avoid over-stripping
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)

def preprocess_text(raw: str, keep_greetings: bool) -> str:
    norm = normalize_transcript(raw)
    if keep_greetings:
        return norm
    return strip_greetings_only(norm)

# -----------------------------
# Schemas
# -----------------------------
class ProblemsOut(BaseModel):
    problems: List[str] = Field(default_factory=list)

class ObservationsOut(BaseModel):
    observations: List[str] = Field(default_factory=list)

class EmotionCogOut(BaseModel):
    emotion_cognition: List[str] = Field(default_factory=list)

class InterventionsOut(BaseModel):
    interventions: List[str] = Field(default_factory=list)

class PatientResponseOut(BaseModel):
    patient_response: List[str] = Field(default_factory=list)

class SOAPOut(BaseModel):
    S: str = ""
    O: str = ""
    A: str = ""
    P: str = ""

class OmahaOut(BaseModel):
    problems: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)

class ReviewOut(BaseModel):
    """Automated QA for coverage, consistency, ontology alignment, and safety."""
    checks: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    suggestions: Optional[Dict[str, Any]] = None

class State(TypedDict, total=False):
    transcript: str
    problems: ProblemsOut
    observations: ObservationsOut
    emotion_cognition: EmotionCogOut
    interventions: InterventionsOut
    patient_response: PatientResponseOut
    soap: SOAPOut
    omaha: OmahaOut
    review: ReviewOut
    final_text: str
    warnings: List[str]
    _llm: Any

# -----------------------------
# LLM factory & shared policy
# -----------------------------
COT_SUFFIX = (
    "Think step-by-step internally, but OUTPUT ONLY valid JSON per the schema. "
    "Use only evidence from the transcript; do not fabricate."
)

def build_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    return ChatOpenAI(model=model, temperature=temperature)

# -----------------------------
# Chain builders
# -----------------------------

def extractor_chain(llm, schema_model: type[BaseModel], task_name: str, key_name: str):
    parser = JsonOutputParser(pydantic_object=schema_model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You extract {task_name} from nurse-patient transcripts. {COT_SUFFIX}"),
        ("human", "Transcript (cleaned):\n{transcript}\n\nReturn only JSON for the schema."),
    ])
    return prompt | llm | parser


def composer_chain(llm):
    parser = JsonOutputParser(pydantic_object=SOAPOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Compose a concise, clinically appropriate SOAP note from the transcript. " + COT_SUFFIX),
        ("human", "Transcript (cleaned):\n{transcript}\n\nReturn JSON with fields S,O,A,P."),
    ])
    return prompt | llm | parser


def omaha_chain(llm):
    parser = JsonOutputParser(pydantic_object=OmahaOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Map content to Omaha System problems and interventions. " + COT_SUFFIX),
        ("human", "Transcript (cleaned):\n{transcript}\n\nReturn JSON with problems[] and interventions[]."),
    ])
    return prompt | llm | parser


def fallback_soap_chain(llm):
    # Simple, robust backup composer
    return composer_chain(llm)


def review_chain(llm):
    parser = JsonOutputParser(pydantic_object=ReviewOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a clinical QA validator for nurse-patient notes. "
         "Validate coverage, consistency, ontology alignment (Omaha), and bias/safety. "
         "Use only the provided content; do not invent facts. "
         "Think step-by-step internally but OUTPUT ONLY JSON matching the schema."),
        ("human",
         "Transcript (cleaned):\n{transcript}\n\n"
         "SOAP JSON:\n{soap}\n\n"
         "Extracts JSON:\n{extracts}\n\n"
         "Omaha JSON:\n{omaha}\n\n"
         "Return JSON with fields: checks, warnings, suggestions (optional)."),
    ])
    return prompt | llm | parser

# -----------------------------
# Utilities
# -----------------------------

def safe_invoke(chain, **kwargs):
    try:
        return chain.invoke(kwargs)
    except Exception as e:
        return None

# -----------------------------
# Nodes
# -----------------------------

def n_problems(state: State) -> State:
    llm = state.get("_llm") or build_llm()
    chain = extractor_chain(llm, ProblemsOut, "patient problems/concerns", "problems")
    out = safe_invoke(chain, transcript=state.get("transcript", ""))
    if out:
        state["problems"] = out
    else:
        state.setdefault("warnings", []).append("Problems extractor failed or returned empty.")
        state["problems"] = ProblemsOut()
    return state


def n_observations(state: State) -> State:
    llm = state.get("_llm") or build_llm()
    chain = extractor_chain(llm, ObservationsOut, "objective observations (vitals/exam)", "observations")
    out = safe_invoke(chain, transcript=state.get("transcript", ""))
    if out:
        state["observations"] = out
    else:
        state.setdefault("warnings", []).append("Observations extractor failed or returned empty.")
        state["observations"] = ObservationsOut()
    return state


def n_emo_cog(state: State) -> State:
    llm = state.get("_llm") or build_llm()
    chain = extractor_chain(llm, EmotionCogOut, "emotion/cognition (affect, orientation)", "emotion_cognition")
    out = safe_invoke(chain, transcript=state.get("transcript", ""))
    if out:
        state["emotion_cognition"] = out
    else:
        state.setdefault("warnings", []).append("Emotion/Cognition extractor failed or returned empty.")
        state["emotion_cognition"] = EmotionCogOut()
    return state


def n_interventions(state: State) -> State:
    llm = state.get("_llm") or build_llm()
    chain = extractor_chain(llm, InterventionsOut, "nursing interventions (education/monitoring)", "interventions")
    out = safe_invoke(chain, transcript=state.get("transcript", ""))
    if out:
        state["interventions"] = out
    else:
        state.setdefault("warnings", []).append("Interventions extractor failed or returned empty.")
        state["interventions"] = InterventionsOut()
    return state


def n_patient_resp(state: State) -> State:
    llm = state.get("_llm") or build_llm()
    chain = extractor_chain(llm, PatientResponseOut, "patient response (understanding/adherence/questions)", "patient_response")
    out = safe_invoke(chain, transcript=state.get("transcript", ""))
    if out:
        state["patient_response"] = out
    else:
        state.setdefault("warnings", []).append("Patient response extractor failed or returned empty.")
        state["patient_response"] = PatientResponseOut()
    return state


def n_soap(state: State) -> State:
    llm = state.get("_llm") or build_llm()
    chain = composer_chain(llm)
    out = safe_invoke(chain, transcript=state.get("transcript", ""))
    if out and any([out.S, out.O, out.A, out.P]):
        state["soap"] = out
    else:
        state.setdefault("warnings", []).append("SOAP composer returned empty; using fallback.")
        fchain = fallback_soap_chain(llm)
        fout = safe_invoke(fchain, transcript=state.get("transcript", "")) or SOAPOut()
        state["soap"] = fout
    return state


def n_omaha(state: State) -> State:
    llm = state.get("_llm") or build_llm()
    chain = omaha_chain(llm)
    out = safe_invoke(chain, transcript=state.get("transcript", ""))
    if out:
        state["omaha"] = out
        # simple validations
        if not out.problems:
            state.setdefault("warnings", []).append("Omaha problems list is empty.")
        if not out.interventions:
            state.setdefault("warnings", []).append("Omaha interventions list is empty.")
    else:
        state.setdefault("warnings", []).append("Omaha mapping failed or returned empty.")
        state["omaha"] = OmahaOut()
    return state


def n_review(state: State) -> State:
    warnings: List[str] = state.get("warnings", []).copy()

    def _lower_set(xs):
        try:
            return {x.strip().lower() for x in (xs or []) if isinstance(x, str)}
        except Exception:
            return set()

    probs = _lower_set(getattr(state.get("problems", None), "problems", []))
    intervs = _lower_set(getattr(state.get("interventions", None), "interventions", []))
    soap = state.get("soap", None)

    s_text = getattr(soap, "S", "") if soap else ""
    o_text = getattr(soap, "O", "") if soap else ""
    a_text = getattr(soap, "A", "") if soap else ""
    p_text = getattr(soap, "P", "") if soap else ""

    # Coverage checks
    for sec_name, sec_val in [("S", s_text), ("O", o_text), ("A", a_text), ("P", p_text)]:
        if not str(sec_val).strip():
            warnings.append(f"SOAP.{sec_name} is empty.")

    # Simple semantic checks
    if "pain" in probs and "pain" not in s_text.lower():
        warnings.append("Problem 'pain' found in extraction but not reflected in SOAP.S.")
    if "shortness of breath" in probs and "shortness of breath" not in (s_text + o_text).lower():
        warnings.append("SOB found in extraction but not reflected in SOAP.S/O.")

    # Ontology presence
    omaha = state.get("omaha", None)
    if omaha:
        if not getattr(omaha, "problems", []):
            warnings.append("Omaha problems list is empty.")
        if not getattr(omaha, "interventions", []):
            warnings.append("Omaha interventions list is empty.")

    # Bias/safety (rule-based placeholder)
    biased_terms = ["non-compliant", "difficult patient"]
    for term in biased_terms:
        if term in s_text.lower() or term in a_text.lower():
            warnings.append(f"Potentially stigmatizing language detected: '{term}'")

    # LLM review
    llm = state.get("_llm") or build_llm()
    rc = review_chain(llm)

    soap_json = soap.model_dump() if hasattr(soap, "model_dump") else {
        "S": s_text, "O": o_text, "A": a_text, "P": p_text
    }
    extracts_json: Dict[str, Any] = {}
    for key in ["problems", "observations", "emotion_cognition", "interventions", "patient_response"]:
        val = state.get(key, None)
        extracts_json[key] = val.model_dump() if hasattr(val, "model_dump") else None
    omaha_json = omaha.model_dump() if hasattr(omaha, "model_dump") else None

    review: Optional[ReviewOut] = safe_invoke(
        rc,
        transcript=state.get("transcript", ""),
        soap=soap_json,
        extracts=extracts_json,
        omaha=omaha_json,
    )

    if review:
        state["review"] = review
        if review.warnings:
            warnings.extend(review.warnings)
    else:
        state["review"] = ReviewOut(checks={}, warnings=[], suggestions=None)

    state["warnings"] = warnings
    return state


def n_finalize(state: State) -> State:
    soap = state.get("soap")
    omaha = state.get("omaha")
    warnings = state.get("warnings", [])

    s = getattr(soap, "S", "") if soap else ""
    o = getattr(soap, "O", "") if soap else ""
    a = getattr(soap, "A", "") if soap else ""
    p = getattr(soap, "P", "") if soap else ""

    omaha_probs = getattr(omaha, "problems", []) if omaha else []
    omaha_intervs = getattr(omaha, "interventions", []) if omaha else []

    lines = [
        "S: " + s,
        "O: " + o,
        "A: " + a,
        "P: " + p,
        "",
        "Problems (Omaha): " + (", ".join(omaha_probs) if omaha_probs else "—"),
        "Interventions (Omaha): " + (", ".join(omaha_intervs) if omaha_intervs else "—"),
    ]

    review = state.get("review")
    if review and getattr(review, "checks", None):
        lines.append("")
        lines.append("QA Checks (summary):")
        for k, v in review.checks.items():
            lines.append(f" - {k}: {v}")

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f" - {w}")

    state["final_text"] = "\n".join(lines) if any([s, o, a, p]) else {}
    return state

# -----------------------------
# Graph builder & runner
# -----------------------------

def build_graph(model: str = "gpt-4o-mini", temperature: float = 0.0):
    llm = build_llm(model=model, temperature=temperature)

    graph = StateGraph(State)

    # Nodes
    graph.add_node("problems", n_problems)
    graph.add_node("observations", n_observations)
    graph.add_node("emo_cog", n_emo_cog)
    graph.add_node("interventions", n_interventions)
    graph.add_node("patient_resp", n_patient_resp)
    graph.add_node("soap", n_soap)
    graph.add_node("omaha", n_omaha)
    graph.add_node("review", n_review)   # NEW
    graph.add_node("finalize", n_finalize)

    # Edges
    graph.add_edge("problems", "observations")
    graph.add_edge("observations", "emo_cog")
    graph.add_edge("emo_cog", "interventions")
    graph.add_edge("interventions", "patient_resp")
    graph.add_edge("patient_resp", "soap")
    graph.add_edge("soap", "omaha")
    graph.add_edge("omaha", "review")     # NEW
    graph.add_edge("review", "finalize")   # NEW

    graph.set_entry_point("problems")
    graph.set_finish_point("finalize")

    app = graph.compile()

    # Provide a small wrapper that seeds the llm into state
    class Runner:
        def invoke(self, init_state: State) -> State:
            init_state = dict(init_state)
            init_state["_llm"] = llm
            return app.invoke(init_state)

    return Runner()

# -----------------------------
# Orchestration helpers
# -----------------------------

def summarize_text(clean_transcript: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> str:
    runner = build_graph(model=model, temperature=temperature)
    result = runner.invoke({"transcript": clean_transcript})

    final_text = result.get("final_text")
    if not final_text:
        # Last-resort fallback to ensure output is not empty
        llm = build_llm(model=model, temperature=temperature)
        fallback = fallback_soap_chain(llm)
        soap = safe_invoke(fallback, transcript=clean_transcript) or SOAPOut()
        lines = [
            "S: " + soap.S,
            "O: " + soap.O,
            "A: " + soap.A,
            "P: " + soap.P,
        ]
        warnings = result.get("warnings", [])
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend([f" - {w}" for w in warnings])
        final_text = "\n".join(lines)
    return final_text

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Agentic Summarization with Review Agent")
    parser.add_argument("--file", type=str, default=None, help="Path to transcript file (default: STDIN)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model id")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--keep-greetings", action="store_true", help="Do not strip greeting/pleasantries")
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    clean = preprocess_text(text, keep_greetings=args.keep_greetings)
    output = summarize_text(clean, model=args.model, temperature=args.temperature)
    print(output)

if __name__ == "__main__":
    main()
