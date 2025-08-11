#!/usr/bin/env python3
"""
LangchainOnly.py
LangChain-only baseline for text-only clinical summarization (no LangGraph).

Output (text only):
S: ...
O: ...
A: ...
P: ...
Problems (Omaha): ...
Interventions (Omaha): ...

Usage:
  export OPENAI_API_KEY="sk-..."; python LangchainOnly.py -f labeled_dialogue.txt
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, TypedDict, Any, Type, TypeVar

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ================================
# Preprocessing (normalize & strip greetings-at-start)
# ================================

SPEAKER_RE = re.compile(r'^\s*SPEAKER_\d+\s*\(([^)]+)\):\s*(.*)$', re.IGNORECASE)

def normalize_transcript(raw: str) -> str:
    """'SPEAKER_01 (Nurse): ...' → 'Nurse: ...' / 'Patient: ...'"""
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

# Pure greetings at the start only
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
CLINICAL_CUES = [
    re.compile(r"\b(pain|fever|cough|breath|breathing|sleep|dizzi|bp|blood pressure|glucose|sugar|wound|medicat|dose|injec|fall|injur|symptom|since|when|monitor|exercise)\b", re.I),
    re.compile(r"\?$"),
    re.compile(r"\bdescribe how\b", re.I),
    re.compile(r"\bhow have you been feeling\b", re.I),
]

def strip_greetings_only(normalized_text: str) -> str:
    """Remove greeting lines ONLY at the start; keep yes/no/okay; stop once clinical talk begins."""
    kept, clinical_started = [], False
    for line in normalized_text.splitlines():
        if ":" not in line:
            continue
        role, text = line.split(":", 1)
        t = text.strip()
        if clinical_started:
            kept.append(f"{role.strip()}: {t}"); continue
        if any(p.search(t) for p in CLINICAL_CUES):
            clinical_started = True; kept.append(f"{role.strip()}: {t}"); continue
        if t.lower() in {"yes", "no", "ok", "okay", "alright"}:
            kept.append(f"{role.strip()}: {t}"); continue
        if any(p.fullmatch(t) for p in GREETING_PATTERNS):
            continue
        kept.append(f"{role.strip()}: {t}")
    return "\n".join(kept)

def preprocess_text(raw: str, keep_greetings: bool = False) -> str:
    norm = normalize_transcript(raw)
    return norm if keep_greetings else strip_greetings_only(norm)

# ================================
# Schemas (internal JSON only)
# ================================

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
    S: str
    O: str
    A: str
    P: str

class OmahaOut(BaseModel):
    problems: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)

# ================================
# LLM + chain builders (LangChain)
# ================================

def build_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """Reads OPENAI_API_KEY from env (recommended)."""
    return ChatOpenAI(model=model, temperature=temperature, api_key="<ADD API KEY>")
USE_COT = True

def cot_suffix() -> str:
    return ("Think step by step internally. Do NOT reveal your chain-of-thought. "
            "Only output valid JSON matching the schema.") if USE_COT else "Only output valid JSON matching the schema."

def _escape_curly(s: str) -> str:
    """Escape { } so LangChain's template engine doesn't treat them as variables."""
    return s.replace("{", "{{").replace("}", "}}")

def extractor_chain(llm: ChatOpenAI, schema_model, task_name: str, key_name: str):
    """
    Create a generic extractor chain that returns JSON (Problems, Observations, etc.).
    Avoid brace-conflicts by injecting format_instructions via .partial(...) AND escaping braces.
    """
    parser = JsonOutputParser(pydantic_object=schema_model)
    fmt = _escape_curly(parser.get_format_instructions())
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a clinical assistant that extracts {task_name} from a patient–nurse conversation. {cot_suffix()}"),
        ("user",
         "Transcript:\n{transcript}\n\n"
         "Rules:\n"
         "- Use only information explicitly stated.\n"
         "- No new facts, vitals, or dates unless stated.\n"
         "- Keep items concise.\n\n"
         "Schema: {format_instructions}\n"
         "Return JSON with key \"{key_name}\".")
    ]).partial(
        format_instructions=fmt,
        key_name=key_name
    )
    return prompt | llm | parser

def composer_chain(llm: ChatOpenAI):
    parser = JsonOutputParser(pydantic_object=SOAPOut)
    fmt = _escape_curly(parser.get_format_instructions())
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Compose a concise SOAP note from the provided lists. "
         "Do NOT add any facts not present in the lists. "
         "Do NOT invent vitals/meds/timelines. "
         + ("Think internally; " if USE_COT else "") + "output JSON for SOAP only."),
        ("user",
         "Problems:\n{problems}\n\n"
         "Observations:\n{observations}\n\n"
         "Emotion/Cognition (verbal):\n{emo_cog}\n\n"
         "Interventions:\n{interventions}\n\n"
         "Patient response:\n{patient_resp}\n\n"
         "Schema: {format_instructions}\n"
         "Style: 1–2 sentences per S/O/A/P; precise clinical language.")
    ]).partial(
        format_instructions=fmt
    )
    return prompt | llm | parser

def omaha_chain(llm: ChatOpenAI):
    parser = JsonOutputParser(pydantic_object=OmahaOut)
    fmt = _escape_curly(parser.get_format_instructions())
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Map to concise Omaha labels for problems and interventions. "
         "Examples of actions: Teaching, Guidance, Surveillance, Treatments. "
         "No codes. " + ("Think internally; " if USE_COT else "") + "output JSON only."),
        ("user",
         "Problems (bullets):\n{problems}\n\n"
         "Interventions (bullets):\n{interventions}\n\n"
         "Schema: {format_instructions}\n"
         "Rules: Only select labels clearly justified by the bullets; keep lists short.")
    ]).partial(
        format_instructions=fmt
    )
    return prompt | llm | parser

def fallback_soap_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a clinical summarizer. Write a SOAP note (S/O/A/P) using only facts explicitly stated "
         "in the transcript. Do NOT invent vitals/meds/dates. "
         "Then add two lines: 'Problems (Omaha): ...' and 'Interventions (Omaha): ...'. "
         + ("Think internally; " if USE_COT else "") + "do not reveal chain-of-thought."),
        ("user",
         "Transcript:\n{transcript}\n\n"
         "Format exactly:\n"
         "S: ...\nO: ...\nA: ...\nP: ...\nProblems (Omaha): ...\nInterventions (Omaha): ...")
    ])
    return prompt | llm

# ================================
# Helpers to normalize outputs
# ================================
T = TypeVar("T", bound=BaseModel)

def ensure_model(obj: Any, cls: Type[T]) -> T:
    """Return obj as Pydantic model instance (handles dicts from some LC versions)."""
    if isinstance(obj, cls):
        return obj
    if isinstance(obj, dict):
        return cls(**obj)
    # last resort (shouldn't be needed)
    return cls.model_validate(obj)

# ================================
# Run sequence (LangChain only)
# ================================
class SeqState(TypedDict, total=False):
    problems: ProblemsOut
    observations: ObservationsOut
    emo_cog: EmotionCogOut
    interventions: InterventionsOut
    patient_resp: PatientResponseOut
    soap: SOAPOut
    omaha: OmahaOut

def summarize_text(transcript: str, model: str = "gpt-4o-mini", temperature: float = 0.0, debug: bool = False) -> str:
    llm = build_llm(model=model, temperature=temperature)

    # Build per-step chains (LangChain runnables)
    ch_problems = extractor_chain(llm, ProblemsOut, "patient-reported problems/symptoms", "problems")
    ch_observ   = extractor_chain(llm, ObservationsOut, "objective observations", "observations")
    ch_emocog   = extractor_chain(llm, EmotionCogOut, "verbalized emotions/cognitive states", "emotion_cognition")
    ch_interv   = extractor_chain(llm, InterventionsOut, "nurse interventions", "interventions")
    ch_presp    = extractor_chain(llm, PatientResponseOut, "patient response", "patient_response")
    ch_soap     = composer_chain(llm)
    ch_omaha    = omaha_chain(llm)
    ch_fallback = fallback_soap_chain(llm)

    state: SeqState = {}
    t0 = time.perf_counter()

    # 1) Extractors
    state["problems"]      = ensure_model(ch_problems.invoke({"transcript": transcript}), ProblemsOut)
    state["observations"]  = ensure_model(ch_observ.invoke({"transcript": transcript}), ObservationsOut)
    state["emo_cog"]       = ensure_model(ch_emocog.invoke({"transcript": transcript}), EmotionCogOut)
    state["interventions"] = ensure_model(ch_interv.invoke({"transcript": transcript}), InterventionsOut)
    state["patient_resp"]  = ensure_model(ch_presp.invoke({"transcript": transcript}), PatientResponseOut)

    # 2) Compose SOAP
    state["soap"] = ensure_model(ch_soap.invoke({
        "problems": state["problems"].problems,
        "observations": state["observations"].observations,
        "emo_cog": state["emo_cog"].emotion_cognition,
        "interventions": state["interventions"].interventions,
        "patient_resp": state["patient_resp"].patient_response,
    }), SOAPOut)

    # 3) Omaha mapping from extracted bullets
    state["omaha"] = ensure_model(ch_omaha.invoke({
        "problems": state["problems"].problems,
        "interventions": state["interventions"].interventions,
    }), OmahaOut)

    # 4) Final text assembly (text-only)
    if state["soap"].S:
        final_text = (
            f"S: {state['soap'].S}\n"
            f"O: {state['soap'].O}\n"
            f"A: {state['soap'].A}\n"
            f"P: {state['soap'].P}\n"
            f"Problems (Omaha): {', '.join(state['omaha'].problems) if state['omaha'].problems else 'None'}\n"
            f"Interventions (Omaha): {', '.join(state['omaha'].interventions) if state['omaha'].interventions else 'None'}"
        )
    else:
        out = ch_fallback.invoke({"transcript": transcript})
        final_text = out.content if hasattr(out, "content") else str(out)

    if debug:
        dt = time.perf_counter() - t0
        print(f"[DEBUG] end-to-end seconds: {dt:.2f}", file=sys.stderr)
        print("[DEBUG] Problems:", state.get("problems").problems, file=sys.stderr)
        print("[DEBUG] Observations:", state.get("observations").observations, file=sys.stderr)
        print("[DEBUG] Emo/Cog:", state.get("emo_cog").emotion_cognition, file=sys.stderr)
        print("[DEBUG] Interventions:", state.get("interventions").interventions, file=sys.stderr)
        print("[DEBUG] Patient response:", state.get("patient_resp").patient_response, file=sys.stderr)

    return final_text

# ================================
# CLI
# ================================
def main():
    ap = argparse.ArgumentParser(description="LangChain-only baseline: SOAP + Omaha (text only).")
    ap.add_argument("--file", "-f", type=str, help="Transcript path; if omitted, read stdin.")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--keep-greetings", action="store_true", help="Keep greetings at start (default strips them).")
    ap.add_argument("--debug", action="store_true", help="Print extracted bullets + timing to stderr.")
    ap.add_argument("--no-cot", action="store_true", help="Disable CoT preamble in prompts.")
    args = ap.parse_args()

    global USE_COT
    if args.no_cot:
        USE_COT = False

    raw = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    clean = preprocess_text(raw, keep_greetings=args.keep_greetings)

    print(summarize_text(clean, model=args.model, temperature=args.temperature, debug=args.debug))

if __name__ == "__main__":
    main()
