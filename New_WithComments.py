#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agentic_summarizer_cli.py
--------------------------------

WHAT THIS DOES (high level):
- Takes a patient–nurse conversation transcript (from file or stdin)
- Preprocesses it (normalize diarization speaker tags; drop opening greetings only)
- Runs a **serial** LangGraph pipeline of extractor heads (Problems, Observations, etc.)
- After **each** extractor, runs a **validator** that drops anything not literally supported
- Composes a SOAP note (S/O/A/P) **only** from validated items
- Maps validated problems/interventions to concise **Omaha** labels
- Builds a human-readable final text block
- Runs a **final hallucination scrub** comparing the summary to the transcript and
  *deleting* any unsupported content (without “improving” or changing meaning)
- If something breaks, uses a **fallback** chain that writes SOAP+Omaha directly

WHY IT’S SAFE:
- Extractor validators: “use only what’s stated” + dedupe + no paraphrases that shift meaning
- Composer prompt: “no-new-facts” instruction
- Final scrubber: removes any unsupported wording that slipped through

RUN:
  pip install langchain langchain-openai pydantic langgraph
  export OPENAI_API_KEY=your_key
  python agentic_summarizer_cli.py --file demo_transcript.txt

NOTE:
- This keeps a **serial** flow. If you later want **parallel fan-out**, only the edges change.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

# LLM + prompt plumbing
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangGraph orchestration
from langgraph.graph import StateGraph, START, END


# ==============================================================
# 0) PREPROCESSING
# --------------------------------------------------------------
# Normalize diarization to "Nurse:" / "Patient:" and remove pure
# greetings *only at the start* (keeps short "yes/no/ok" utterances).
# ==============================================================

SPEAKER_RE = re.compile(r'^\s*SPEAKER_\d+\s*\(([^)]+)\):\s*(.*)$', re.IGNORECASE)

def normalize_transcript(raw: str) -> str:
    """Convert 'SPEAKER_01 (Nurse): ...' → 'Nurse: ...' / 'Patient: ...'."""
    out: List[str] = []
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
    """Drop greetings only at the start, stop once clinical talk begins. Preserve 'yes/no/ok'."""
    kept: List[str] = []
    clinical_started = False
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
# 1) SCHEMAS (Pydantic)
# --------------------------------------------------------------
# Each extractor returns one of these. Validators return ValidatedOut.
# ==============================================================

# Core heads
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

# SOAP object
class SOAPOut(BaseModel):
    S: str
    O: str
    A: str
    P: str

# Omaha mapping object
class OmahaOut(BaseModel):
    problems: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)

# Extended clinically relevant heads
class MedicationsOut(BaseModel):
    # Example item: {"name":"albuterol","dose":"2 puffs","route":"INH","freq":"q4h PRN","changes":"…","adherence_issue":True}
    meds: List[Dict[str, Any]] = Field(default_factory=list)

class AllergiesOut(BaseModel):
    # Example item: {"substance":"penicillin","reaction":"rash","severity":"?","when":"?"}
    allergies: List[Dict[str, Any]] = Field(default_factory=list)

class VitalsOut(BaseModel):
    # Example item: {"type":"BP","value":"150/90","unit":None,"when":"today","source":"nurse"}
    vitals: List[Dict[str, Any]] = Field(default_factory=list)

class HistoryOut(BaseModel):
    conditions: List[str] = Field(default_factory=list)

class SDOHOut(BaseModel):
    factors: List[str] = Field(default_factory=list)

class FunctionalOut(BaseModel):
    adls: List[str] = Field(default_factory=list)
    iadls: List[str] = Field(default_factory=list)

class RiskSafetyOut(BaseModel):
    risks: List[str] = Field(default_factory=list)

class EducationOut(BaseModel):
    taught: List[str] = Field(default_factory=list)
    teachback_ok: Optional[bool] = None
    barriers: List[str] = Field(default_factory=list)

class FollowupsOut(BaseModel):
    # Example item: {"task":"pulmonology follow-up","who":"clinic","when_due":"next week"}
    actions: List[Dict[str, Any]] = Field(default_factory=list)

# Validator common output
class ValidatedOut(BaseModel):
    kept: List[Any] = Field(default_factory=list)     # kept items (same structure as candidates)
    removed: List[Any] = Field(default_factory=list)  # dropped items
    notes: List[str] = Field(default_factory=list)


# ==============================================================
# 2) GRAPH STATE
# --------------------------------------------------------------
# Shared across nodes; LangGraph merges dict updates between nodes.
# ==============================================================

class State(TypedDict, total=False):
    transcript: str

    # Core heads (raw + validated)
    problems: ProblemsOut
    v_problems: ValidatedOut
    observations: ObservationsOut
    v_observations: ValidatedOut
    emo_cog: EmotionCogOut
    v_emo_cog: ValidatedOut
    interventions: InterventionsOut
    v_interventions: ValidatedOut
    patient_resp: PatientResponseOut
    v_patient_resp: ValidatedOut

    # Extended heads (raw + validated)
    medications: MedicationsOut
    v_medications: ValidatedOut
    allergies: AllergiesOut
    v_allergies: ValidatedOut
    vitals: VitalsOut
    v_vitals: ValidatedOut
    history: HistoryOut
    v_history: ValidatedOut
    sdoh: SDOHOut
    v_sdoh: ValidatedOut
    functional: FunctionalOut
    v_functional: ValidatedOut
    risksafety: RiskSafetyOut
    v_risksafety: ValidatedOut
    education: EducationOut
    v_education: ValidatedOut
    followups: FollowupsOut
    v_followups: ValidatedOut

    # Downstream
    soap: SOAPOut
    omaha: OmahaOut
    final_text: str
    warnings: List[str]


# ==============================================================
# 3) CHAIN FACTORIES (LLM prompts)
# --------------------------------------------------------------
# Build extractors, validators, composer, omaha mapper, fallback,
# and final hallucination scrubber.
# ==============================================================

def build_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Build a chat model. Reads OPENAI_API_KEY from environment.
    """
    return ChatOpenAI(model=model, temperature=temperature)

# CoT policy: think internally, but never reveal
COT_SUFFIX = (
    "Think step by step internally. Do NOT reveal your chain-of-thought. "
    "Only output valid JSON matching the schema."
)

def extractor_chain(llm, schema_model, task_name: str, key_name: str):
    """
    Generic extractor that outputs JSON constrained by `schema_model`.
    """
    parser = JsonOutputParser(pydantic_object=schema_model)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are a clinical assistant that extracts {task_name} from a patient–nurse conversation. {COT_SUFFIX}"),
        ("user",
         "Transcript:\n{transcript}\n\n"
         "Rules:\n"
         "- Use only information explicitly stated (verbatim support).\n"
         "- No new facts, vitals, or dates unless stated.\n"
         "- Keep items concise.\n\n"
         f"Schema: {parser.get_format_instructions()}\n"
         f"Return JSON with key '{key_name}'.")
    ])
    return prompt | llm | parser

def validator_chain(llm, task_name: str):
    """
    Validates extractor items against the transcript: keep only literally supported,
    dedupe, and compress wording without changing meaning.
    """
    parser = JsonOutputParser(pydantic_object=ValidatedOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You validate extracted {task_name} against the transcript. "
         "Keep ONLY items explicitly supported by the transcript. "
         "If two items are duplicates, keep one concise form and remove the rest. "
         "Do not paraphrase in ways that change meaning. "
         "Think silently; output JSON only."),
        ("user",
         "Transcript:\n{transcript}\n\n"
         f"Candidate {task_name} (list or object):\n{{candidates}}\n\n"
         "Rules:\n"
         "- Each kept item must be literally supportable by one or more transcript lines.\n"
         "- Remove items that are implied but not stated.\n"
         "- Prefer exact phrasing or neutral compression.\n"
         f"Schema: {parser.get_format_instructions()}")
    ])
    return prompt | llm | parser

def composer_chain(llm):
    """
    LLM composer for SOAP. Uses only validated lists; instructed to add no new facts.
    """
    parser = JsonOutputParser(pydantic_object=SOAPOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Compose a concise SOAP note from the provided validated lists. "
         "Do NOT add any facts not present in these lists. "
         "Do NOT invent vitals/meds/timelines. "
         "Think internally; output JSON for SOAP only."),
        ("user",
         "Validated Problems:\n{v_problems}\n\n"
         "Validated Observations:\n{v_observations}\n\n"
         "Validated Emotion/Cognition:\n{v_emo_cog}\n\n"
         "Validated Interventions:\n{v_interventions}\n\n"
         "Validated Patient Response:\n{v_patient_resp}\n\n"
         "Validated Medications:\n{v_medications}\n\n"
         "Validated Allergies:\n{v_allergies}\n\n"
         "Validated Vitals:\n{v_vitals}\n\n"
         "Validated History:\n{v_history}\n\n"
         "Validated SDOH:\n{v_sdoh}\n\n"
         "Validated Functional:\n{v_functional}\n\n"
         "Validated Risk/Safety:\n{v_risksafety}\n\n"
         "Validated Education/Teach-back:\n{v_education}\n\n"
         "Validated Follow-ups:\n{v_followups}\n\n"
         f"Schema: {parser.get_format_instructions()}\n"
         "Style: 1–2 sentences per S/O/A/P; precise clinical language.")
    ])
    return prompt | llm | parser

def omaha_chain(llm):
    """
    Map validated problems/interventions to concise Omaha labels (no codes).
    """
    parser = JsonOutputParser(pydantic_object=OmahaOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Map to concise Omaha labels for problems and interventions. "
         "Examples of actions: Teaching, Guidance, Surveillance, Treatments. "
         "No codes. Think internally; output JSON only."),
        ("user",
         "Problems (validated bullets):\n{problems}\n\n"
         "Interventions (validated bullets):\n{interventions}\n\n"
         f"Schema: {parser.get_format_instructions()}\n"
         "Rules: Only select labels clearly justified by the bullets; keep lists short.")
    ])
    return prompt | llm | parser

def fallback_soap_chain(llm):
    """
    Backup: produce text-only SOAP+Omaha directly from transcript if structured path fails.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a clinical summarizer. Write a SOAP note (S/O/A/P) using only facts explicitly stated "
         "in the transcript. Do NOT invent vitals/meds/dates. "
         "Then add two lines: 'Problems (Omaha): ...' and 'Interventions (Omaha): ...'. "
         "Do NOT reveal chain-of-thought."),
        ("user",
         "Transcript:\n{transcript}\n\n"
         "Format exactly:\n"
         "S: ...\nO: ...\nA: ...\nP: ...\nProblems (Omaha): ...\nInterventions (Omaha): ...")
    ])
    return prompt | llm

def factuality_guard_chain(llm):
    """
    Final hallucination scrub: compare final summary vs transcript; delete unsupported content.
    Keeps headings and order exactly.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict factuality auditor. "
         "Revise the summary so it contains ONLY facts stated in the transcript. "
         "Delete unsupported claims. Do NOT add wording that changes meaning. "
         "Keep the exact required format and headings."),
        ("user",
         "Transcript:\n{transcript}\n\n"
         "Current summary (must keep headings and order exactly):\n{summary}\n\n"
         "Instructions:\n"
         "- Remove any content not literally supported by the transcript.\n"
         "- If a sentence mixes supported and unsupported content, keep the supported part and delete the rest.\n"
         "- Do NOT add synonyms that change meaning.\n"
         "- Keep headings exactly: S:, O:, A:, P:, Problems (Omaha):, Interventions (Omaha):")
    ])
    return prompt | llm


# ==============================================================
# 4) NODES (execution helpers)
# --------------------------------------------------------------
# `safe_invoke` prevents the whole run from failing; we log warnings.
# `n_extract` and `n_validate` are generic node bodies used by many heads.
# ==============================================================

def safe_invoke(chain, payload: Dict[str, Any], empty_value: Any):
    """
    Call a chain and return (result, warn_string|None).
    If the chain errors, return (empty_value, warning).
    """
    try:
        return chain.invoke(payload), None
    except Exception as e:
        return empty_value, f"{type(e).__name__}: {e}"

def n_extract(state: State, chain, dst_key: str, empty_obj: BaseModel):
    """Run an extractor chain; stash result at `dst_key` (e.g., 'problems')."""
    out, warn = safe_invoke(chain, {"transcript": state["transcript"]}, empty_obj)
    updates: Dict[str, Any] = {dst_key: out}
    if warn:
        updates.setdefault("warnings", []).append(f"{dst_key}_node: {warn}")
    return updates

def get_items_for_validator(state: State, src_key: str) -> Any:
    """
    Return the list/object of candidates for a given extractor key (varies by schema).
    """
    obj = state.get(src_key)
    if obj is None:
        return []
    if isinstance(obj, ProblemsOut):
        return obj.problems
    if isinstance(obj, ObservationsOut):
        return obj.observations
    if isinstance(obj, EmotionCogOut):
        return obj.emotion_cognition
    if isinstance(obj, InterventionsOut):
        return obj.interventions
    if isinstance(obj, PatientResponseOut):
        return obj.patient_response
    if isinstance(obj, MedicationsOut):
        return obj.meds
    if isinstance(obj, AllergiesOut):
        return obj.allergies
    if isinstance(obj, VitalsOut):
        return obj.vitals
    if isinstance(obj, HistoryOut):
        return obj.conditions
    if isinstance(obj, SDOHOut):
        return obj.factors
    if isinstance(obj, FunctionalOut):
        return {"adls": obj.adls, "iadls": obj.iadls}  # preserve structure
    if isinstance(obj, RiskSafetyOut):
        return obj.risks
    if isinstance(obj, EducationOut):
        return {"taught": obj.taught, "teachback_ok": obj.teachback_ok, "barriers": obj.barriers}
    if isinstance(obj, FollowupsOut):
        return obj.actions
    return []

def n_validate(state: State, chain, src_key: str, dst_key: str):
    """Run a validator chain on previously extracted `src_key`; store as `dst_key` (e.g., 'v_problems')."""
    candidates = get_items_for_validator(state, src_key)
    out, warn = safe_invoke(chain, {"transcript": state["transcript"], "candidates": candidates}, ValidatedOut())
    updates: Dict[str, Any] = {dst_key: out}
    if warn:
        updates.setdefault("warnings", []).append(f"{dst_key}_node: {warn}")
    return updates

def n_soap(state: State, chain):
    """Compose SOAP from validated lists (LLM composer)."""
    out, warn = safe_invoke(chain, {
        "v_problems": state.get("v_problems", ValidatedOut()).kept,
        "v_observations": state.get("v_observations", ValidatedOut()).kept,
        "v_emo_cog": state.get("v_emo_cog", ValidatedOut()).kept,
        "v_interventions": state.get("v_interventions", ValidatedOut()).kept,
        "v_patient_resp": state.get("v_patient_resp", ValidatedOut()).kept,
        "v_medications": state.get("v_medications", ValidatedOut()).kept,
        "v_allergies": state.get("v_allergies", ValidatedOut()).kept,
        "v_vitals": state.get("v_vitals", ValidatedOut()).kept,
        "v_history": state.get("v_history", ValidatedOut()).kept,
        "v_sdoh": state.get("v_sdoh", ValidatedOut()).kept,
        "v_functional": state.get("v_functional", ValidatedOut()).kept,
        "v_risksafety": state.get("v_risksafety", ValidatedOut()).kept,
        "v_education": state.get("v_education", ValidatedOut()).kept,
        "v_followups": state.get("v_followups", ValidatedOut()).kept,
    }, SOAPOut(S="", O="", A="", P=""))
    updates: Dict[str, Any] = {"soap": out}
    if warn:
        updates.setdefault("warnings", []).append(f"soap_node: {warn}")
    return updates

def n_omaha(state: State, chain):
    """Map validated problems/interventions to concise Omaha labels."""
    out, warn = safe_invoke(chain, {
        "problems": state.get("v_problems", ValidatedOut()).kept,
        "interventions": state.get("v_interventions", ValidatedOut()).kept,
    }, OmahaOut())
    updates: Dict[str, Any] = {"omaha": out}
    if warn:
        updates.setdefault("warnings", []).append(f"omaha_node: {warn}")
    return updates

def n_finalize(state: State) -> State:
    """
    Build final human-readable text block.
    Uses placeholders if SOAP sections are empty (keeps strict no-hallucination policy).
    """
    soap: SOAPOut = state.get("soap", SOAPOut(S="", O="", A="", P=""))  # type: ignore
    omaha: OmahaOut = state.get("omaha", OmahaOut())                    # type: ignore

    s = soap.S.strip() if soap.S and soap.S.strip() else "No subjective information extracted."
    o = soap.O.strip() if soap.O and soap.O.strip() else "No objective observations extracted."
    a = soap.A.strip() if soap.A and soap.A.strip() else "No assessment available."
    p = soap.P.strip() if soap.P and soap.P.strip() else "No plan documented."

    final_text = (
        f"S: {s}\n"
        f"O: {o}\n"
        f"A: {a}\n"
        f"P: {p}\n"
        f"Problems (Omaha): {', '.join(omaha.problems) if omaha.problems else 'None'}\n"
        f"Interventions (Omaha): {', '.join(omaha.interventions) if omaha.interventions else 'None'}"
    )
    return {"final_text": final_text}

def n_final_review(state: State, chain):
    """
    Final hallucination scrub: compare {final_text} vs transcript and delete unsupported material.
    """
    current = state.get("final_text", "")
    out, warn = safe_invoke(chain, {"transcript": state["transcript"], "summary": current}, "")
    updates: Dict[str, Any] = {}

    # Handle either plain string or AIMessage from LangChain
    txt = out
    if hasattr(out, "content"):
        txt = getattr(out, "content", "")

    if isinstance(txt, str) and txt.strip():
        updates["final_text"] = txt.strip()
    else:
        updates.setdefault("warnings", []).append("final_review: empty or invalid output")

    if warn:
        updates.setdefault("warnings", []).append(f"final_review: {warn}")
    return updates


# ==============================================================
# 5) GRAPH BUILDER (SERIAL)
# --------------------------------------------------------------
# START → core extractors+validators → extended extractors+validators
# → SOAP → Omaha → Finalize → Factuality scrub → END
# ==============================================================

def build_graph(model: str = "gpt-4o-mini", temperature: float = 0.0):
    llm = build_llm(model=model, temperature=temperature)

    # --- extractors (core) ---
    problems_node      = extractor_chain(llm, ProblemsOut,       "patient-reported problems/symptoms", "problems")
    observations_node  = extractor_chain(llm, ObservationsOut,   "objective observations", "observations")
    emo_cog_node       = extractor_chain(llm, EmotionCogOut,     "verbalized emotions/cognitive states", "emotion_cognition")
    interventions_node = extractor_chain(llm, InterventionsOut,  "nurse interventions", "interventions")
    patient_resp_node  = extractor_chain(llm, PatientResponseOut,"patient response", "patient_response")

    # --- extractors (extended) ---
    medications_node   = extractor_chain(llm, MedicationsOut,    "medications (name, dose, route, frequency, changes, adherence issues, OTC)", "meds")
    allergies_node     = extractor_chain(llm, AllergiesOut,      "allergies and adverse reactions", "allergies")
    vitals_node        = extractor_chain(llm, VitalsOut,         "vitals/measures mentioned (type, value, unit, when, source)", "vitals")
    history_node       = extractor_chain(llm, HistoryOut,        "past history and comorbidities", "conditions")
    sdoh_node          = extractor_chain(llm, SDOHOut,           "social determinants impacting care", "factors")
    functional_node    = extractor_chain(llm, FunctionalOut,     "functional status (ADLs/IADLs)", "adls")  # parser tolerates extra keys
    risksafety_node    = extractor_chain(llm, RiskSafetyOut,     "risk and safety factors", "risks")
    education_node     = extractor_chain(llm, EducationOut,      "education topics, teach-back, barriers", "taught")
    followups_node     = extractor_chain(llm, FollowupsOut,      "follow-ups/tasks (task, who, when_due)", "actions")

    # --- validators (all heads) ---
    v_problems_node      = validator_chain(llm, "problems")
    v_observations_node  = validator_chain(llm, "observations")
    v_emo_cog_node       = validator_chain(llm, "emotion/cognition")
    v_interventions_node = validator_chain(llm, "interventions")
    v_patient_resp_node  = validator_chain(llm, "patient response")
    v_medications_node   = validator_chain(llm, "medications")
    v_allergies_node     = validator_chain(llm, "allergies")
    v_vitals_node        = validator_chain(llm, "vitals")
    v_history_node       = validator_chain(llm, "history")
    v_sdoh_node          = validator_chain(llm, "SDOH")
    v_functional_node    = validator_chain(llm, "functional status (ADLs/IADLs)")
    v_risksafety_node    = validator_chain(llm, "risk/safety")
    v_education_node     = validator_chain(llm, "education/teach-back")
    v_followups_node     = validator_chain(llm, "follow-ups")

    # --- composer / mapper / scrubber ---
    soap_node  = composer_chain(llm)
    omaha_node = omaha_chain(llm)
    fact_guard = factuality_guard_chain(llm)

    # Build the LangGraph
    g = StateGraph(State)

    # Core heads
    g.add_node("problems",          lambda s: n_extract(s, problems_node,      "problems",      ProblemsOut()))
    g.add_node("v_problems",        lambda s: n_validate(s, v_problems_node,   "problems",      "v_problems"))

    g.add_node("observations",      lambda s: n_extract(s, observations_node,  "observations",  ObservationsOut()))
    g.add_node("v_observations",    lambda s: n_validate(s, v_observations_node,"observations", "v_observations"))

    g.add_node("emo_cog",           lambda s: n_extract(s, emo_cog_node,       "emo_cog",       EmotionCogOut()))
    g.add_node("v_emo_cog",         lambda s: n_validate(s, v_emo_cog_node,    "emo_cog",       "v_emo_cog"))

    g.add_node("interventions",     lambda s: n_extract(s, interventions_node, "interventions", InterventionsOut()))
    g.add_node("v_interventions",   lambda s: n_validate(s, v_interventions_node,"interventions","v_interventions"))

    g.add_node("patient_resp",      lambda s: n_extract(s, patient_resp_node,  "patient_resp",  PatientResponseOut()))
    g.add_node("v_patient_resp",    lambda s: n_validate(s, v_patient_resp_node,"patient_resp", "v_patient_resp"))

    # Extended heads
    g.add_node("medications",       lambda s: n_extract(s, medications_node,   "medications",   MedicationsOut()))
    g.add_node("v_medications",     lambda s: n_validate(s, v_medications_node,"medications",   "v_medications"))

    g.add_node("allergies",         lambda s: n_extract(s, allergies_node,     "allergies",     AllergiesOut()))
    g.add_node("v_allergies",       lambda s: n_validate(s, v_allergies_node,  "allergies",     "v_allergies"))

    g.add_node("vitals",            lambda s: n_extract(s, vitals_node,        "vitals",        VitalsOut()))
    g.add_node("v_vitals",          lambda s: n_validate(s, v_vitals_node,     "vitals",        "v_vitals"))

    g.add_node("history",           lambda s: n_extract(s, history_node,       "history",       HistoryOut()))
    g.add_node("v_history",         lambda s: n_validate(s, v_history_node,    "history",       "v_history"))

    g.add_node("sdoh",              lambda s: n_extract(s, sdoh_node,          "sdoh",          SDOHOut()))
    g.add_node("v_sdoh",            lambda s: n_validate(s, v_sdoh_node,       "sdoh",          "v_sdoh"))

    g.add_node("functional",        lambda s: n_extract(s, functional_node,    "functional",    FunctionalOut()))
    g.add_node("v_functional",      lambda s: n_validate(s, v_functional_node, "functional",    "v_functional"))

    g.add_node("risksafety",        lambda s: n_extract(s, risksafety_node,    "risksafety",    RiskSafetyOut()))
    g.add_node("v_risksafety",      lambda s: n_validate(s, v_risksafety_node, "risksafety",    "v_risksafety"))

    g.add_node("education",         lambda s: n_extract(s, education_node,     "education",     EducationOut()))
    g.add_node("v_education",       lambda s: n_validate(s, v_education_node,  "education",     "v_education"))

    g.add_node("followups",         lambda s: n_extract(s, followups_node,     "followups",     FollowupsOut()))
    g.add_node("v_followups",       lambda s: n_validate(s, v_followups_node,  "followups",     "v_followups"))

    # Downstream steps
    g.add_node("soap",              lambda s: n_soap(s, soap_node))
    g.add_node("omaha",             lambda s: n_omaha(s, omaha_node))
    g.add_node("finalize",          n_finalize)
    g.add_node("final_review",      lambda s: n_final_review(s, fact_guard))

    # SERIAL EDGES
    g.add_edge(START, "problems")
    g.add_edge("problems", "v_problems")
    g.add_edge("v_problems", "observations")
    g.add_edge("observations", "v_observations")
    g.add_edge("v_observations", "emo_cog")
    g.add_edge("emo_cog", "v_emo_cog")
    g.add_edge("v_emo_cog", "interventions")
    g.add_edge("interventions", "v_interventions")
    g.add_edge("v_interventions", "patient_resp")
    g.add_edge("patient_resp", "v_patient_resp")

    # Extended heads after the core
    g.add_edge("v_patient_resp", "medications")
    g.add_edge("medications", "v_medications")
    g.add_edge("v_medications", "allergies")
    g.add_edge("allergies", "v_allergies")
    g.add_edge("v_allergies", "vitals")
    g.add_edge("vitals", "v_vitals")
    g.add_edge("v_vitals", "history")
    g.add_edge("history", "v_history")
    g.add_edge("v_history", "sdoh")
    g.add_edge("sdoh", "v_sdoh")
    g.add_edge("v_sdoh", "functional")
    g.add_edge("functional", "v_functional")
    g.add_edge("v_functional", "risksafety")
    g.add_edge("risksafety", "v_risksafety")
    g.add_edge("v_risksafety", "education")
    g.add_edge("education", "v_education")
    g.add_edge("v_education", "followups")
    g.add_edge("followups", "v_followups")

    # Compose → Omaha → Finalize → Final scrub → END
    g.add_edge("v_followups", "soap")
    g.add_edge("soap", "omaha")
    g.add_edge("omaha", "finalize")
    g.add_edge("finalize", "final_review")
    g.add_edge("final_review", END)

    return g.compile(), llm


# ==============================================================
# 6) RUNNER
# --------------------------------------------------------------
# Orchestrate the graph, apply fallback if needed, return text.
# ==============================================================

def summarize_text(transcript: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> str:
    app, llm = build_graph(model=model, temperature=temperature)
    state = app.invoke({"transcript": transcript})
    final_text = state.get("final_text")
    if final_text:
        return final_text

    # Fallback: robust single-pass SOAP + Omaha directly from transcript
    fb = fallback_soap_chain(llm)
    out = fb.invoke({"transcript": transcript})
    return getattr(out, "content", str(out))


# ==============================================================
# 7) CLI
# --------------------------------------------------------------
# Reads input, preprocesses, summarizes, prints result.
# ==============================================================

def main():
    ap = argparse.ArgumentParser(description="Summarize patient–nurse transcript to SOAP + Omaha (serial, validated, hallucination-scrubbed, extended heads).")
    ap.add_argument("--file", "-f", type=str, help="Path to transcript file. If omitted, read from stdin.")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name (default: gpt-4o-mini)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    ap.add_argument("--keep-greetings", action="store_true", help="Keep greetings at start (default: strip)")
    args = ap.parse_args()

    # Read input
    if args.file:
        raw = Path(args.file).read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()

    # Preprocess
    clean = preprocess_text(raw, keep_greetings=args.keep_greetings)

    # Summarize
    text = summarize_text(clean, model=args.model, temperature=args.temperature)
    print(text)

if __name__ == "__main__":
    main()
