#!/usr/bin/env python3
"""
agentic_summarizer_cli.py
--------------------------------
Serial extractor pipeline with per-extractor validators + final hallucination scrub.
Expanded with additional clinically relevant heads (meds, allergies, vitals, history, SDOH, functional,
risk/safety, education/teach-back, follow-ups).

Outputs (text only):
S: ...
O: ...
A: ...
P: ...
Problems (Omaha): ...
Interventions (Omaha): ...

Usage:
  python agentic_summarizer_cli.py --file path/to/transcript.txt
  cat transcript.txt | python agentic_summarizer_cli.py

Options:
  --model gpt-4o-mini
  --temperature 0.0
  --keep-greetings
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

# ================================
# 0) PREPROCESSING (text only)
# ================================

SPEAKER_RE = re.compile(r'^\s*SPEAKER_\d+\s*\(([^)]+)\):\s*(.*)$', re.IGNORECASE)

def normalize_transcript(raw: str) -> str:
    """
    'SPEAKER_01 (Nurse): ...' → 'Nurse: ...' / 'Patient: ...'
    """
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
    norm = normalize_transcript(raw)
    return norm if keep_greetings else strip_greetings_only(norm)

# =========================================
# 1) Pydantic Schemas (internal JSON only)
# =========================================

# Original heads
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

# New clinically relevant heads
class MedicationsOut(BaseModel):
    meds: List[Dict[str, Any]] = Field(default_factory=list)  # {name, dose?, route?, freq?, purpose?, changes?, adherence_issue?}

class AllergiesOut(BaseModel):
    allergies: List[Dict[str, Any]] = Field(default_factory=list)  # {substance, reaction?, severity?}

class VitalsOut(BaseModel):
    vitals: List[Dict[str, Any]] = Field(default_factory=list)  # {type, value, unit, when?, source?}

class HistoryOut(BaseModel):
    conditions: List[str] = Field(default_factory=list)  # chronic conditions, surgeries (text list)

class SDOHOut(BaseModel):
    factors: List[str] = Field(default_factory=list)  # e.g., transportation barrier, food insecurity

class FunctionalOut(BaseModel):
    adls: List[str] = Field(default_factory=list)
    iadls: List[str] = Field(default_factory=list)

class RiskSafetyOut(BaseModel):
    risks: List[str] = Field(default_factory=list)  # e.g., high fall risk, med mismanagement, home hazards

class EducationOut(BaseModel):
    taught: List[str] = Field(default_factory=list)
    teachback_ok: Optional[bool] = None
    barriers: List[str] = Field(default_factory=list)

class FollowupsOut(BaseModel):
    actions: List[Dict[str, Any]] = Field(default_factory=list)  # {task, who?, when_due?}

# Shared validator result
class ValidatedOut(BaseModel):
    kept: List[Any] = Field(default_factory=list)     # Items retained, same structural type as candidates
    removed: List[Any] = Field(default_factory=list)  # Items dropped
    notes: List[str] = Field(default_factory=list)

# ================================
# 2) LangGraph State
# ================================

class State(TypedDict, total=False):
    transcript: str
    # originals
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
    # new heads
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

    # downstream
    soap: SOAPOut
    omaha: OmahaOut
    final_text: str
    warnings: List[str]

# =================================
# 3) LLM builder & chain factories
# =================================

def build_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    # ChatOpenAI reads OPENAI_API_KEY from env
    return ChatOpenAI(model=model, temperature=temperature)

COT_SUFFIX = (
    "Think step by step internally. Do NOT reveal your chain-of-thought. "
    "Only output valid JSON matching the schema."
)

def extractor_chain(llm, schema_model, task_name: str, key_name: str):
    parser = JsonOutputParser(pydantic_object=schema_model)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are a clinical assistant that extracts {task_name} from a patient–nurse conversation. {COT_SUFFIX}"),
        ("user",
         "Transcript:\n{transcript}\n\n"
         "Rules:\n"
         "- Use only information explicitly stated.\n"
         "- No new facts, vitals, or dates unless stated.\n"
         "- Keep items concise.\n\n"
         f"Schema: {parser.get_format_instructions()}\n"
         f"Return JSON with key '{key_name}'.")
    ])
    return prompt | llm | parser

def validator_chain(llm, task_name: str):
    parser = JsonOutputParser(pydantic_object=ValidatedOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You validate extracted {task_name} against the transcript. "
         "Keep ONLY items explicitly supported by the transcript. "
         "No paraphrased facts that change meaning. "
         "If two items are duplicates, keep one concise form and remove the rest. "
         "Think silently; output JSON only."),
        ("user",
         "Transcript:\n{transcript}\n\n"
         f"Candidate {task_name} (list):\n{{candidates}}\n\n"
         "Rules:\n"
         "- For each kept item, it must be literally supportable by one or more lines.\n"
         "- Remove items that are implied but not stated.\n"
         "- Prefer exact phrasing from the transcript or neutral compression.\n"
         f"Schema: {parser.get_format_instructions()}")
    ])
    return prompt | llm | parser

def composer_chain(llm):
    """
    Compose SOAP from VALIDATED lists only.
    We pass in the validated outputs so the model has rich context while staying no-new-facts.
    """
    parser = JsonOutputParser(pydantic_object=SOAPOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Compose a concise SOAP note from the provided validated lists. "
         "Do NOT add any facts not present in these lists. "
         "Do NOT invent vitals/meds/timelines. "
         "Think internally; output JSON for SOAP only."),
        ("user",
         # Core validated heads
         "Validated Problems:\n{v_problems}\n\n"
         "Validated Observations:\n{v_observations}\n\n"
         "Validated Emotion/Cognition:\n{v_emo_cog}\n\n"
         "Validated Interventions:\n{v_interventions}\n\n"
         "Validated Patient Response:\n{v_patient_resp}\n\n"
         # Supplemental validated heads
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
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict factuality auditor. "
         "Revise the summary so it contains ONLY facts stated in the transcript. "
         "Delete unsupported claims. Do NOT add new wording that changes meaning. "
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

# ================================
# 4) Nodes (with resilience)
# ================================

def safe_invoke(chain, payload, empty_value):
    try:
        return chain.invoke(payload), None
    except Exception as e:
        return empty_value, f"{type(e).__name__}: {e}"

# ---- generic helpers for extract → validate ----

def n_extract(state: State, chain, dst_key: str, empty_obj: BaseModel):
    out, warn = safe_invoke(chain, {"transcript": state["transcript"]}, empty_obj)
    updates = {dst_key: out}
    if warn:
        updates.setdefault("warnings", []).append(f"{dst_key}_node: {warn}")
    return updates

def get_items_for_validator(state: State, src_key: str) -> Any:
    """
    Returns the list of 'candidates' for a given extractor key.
    Handles different schema attribute names.
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
        return {"adls": obj.adls, "iadls": obj.iadls}  # keep structure
    if isinstance(obj, RiskSafetyOut):
        return obj.risks
    if isinstance(obj, EducationOut):
        return {"taught": obj.taught, "teachback_ok": obj.teachback_ok, "barriers": obj.barriers}
    if isinstance(obj, FollowupsOut):
        return obj.actions
    return []

def n_validate(state: State, chain, src_key: str, dst_key: str):
    candidates = get_items_for_validator(state, src_key)
    out, warn = safe_invoke(chain, {"transcript": state["transcript"], "candidates": candidates}, ValidatedOut())
    updates = {dst_key: out}
    if warn:
        updates.setdefault("warnings", []).append(f"{dst_key}_node: {warn}")
    return updates

# ---- composer / omaha / finalize ----

def n_soap(state: State, chain):
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
    updates = {"soap": out}
    if warn:
        updates.setdefault("warnings", []).append(f"soap_node: {warn}")
    return updates

def n_omaha(state: State, chain):
    out, warn = safe_invoke(chain, {
        "problems": state.get("v_problems", ValidatedOut()).kept,
        "interventions": state.get("v_interventions", ValidatedOut()).kept,
    }, OmahaOut())
    updates = {"omaha": out}
    if warn:
        updates.setdefault("warnings", []).append(f"omaha_node: {warn}")
    return updates

def n_finalize(state: State) -> State:
    soap = state.get("soap", SOAPOut(S="", O="", A="", P=""))
    omaha = state.get("omaha", OmahaOut())
    # Placeholder-friendly finalization (safe & readable)
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
    current = state.get("final_text", "")
    out, warn = safe_invoke(chain, {"transcript": state["transcript"], "summary": current}, "")
    updates = {}
    if isinstance(out, str) and out.strip():
        updates["final_text"] = out.strip()
    else:
        updates.setdefault("warnings", []).append("final_review: empty or invalid output")
    if warn:
        updates.setdefault("warnings", []).append(f"final_review: {warn}")
    return updates

# ================================
# 5) Graph builder (SERIAL)
# ================================

def build_graph(model: str = "gpt-4o-mini", temperature: float = 0.0):
    llm = build_llm(model=model, temperature=temperature)

    # --- extractors (original) ---
    problems_node      = extractor_chain(llm, ProblemsOut,       "patient-reported problems/symptoms", "problems")
    observations_node  = extractor_chain(llm, ObservationsOut,   "objective observations", "observations")
    emo_cog_node       = extractor_chain(llm, EmotionCogOut,     "verbalized emotions/cognitive states", "emotion_cognition")
    interventions_node = extractor_chain(llm, InterventionsOut,  "nurse interventions", "interventions")
    patient_resp_node  = extractor_chain(llm, PatientResponseOut,"patient response", "patient_response")

    # --- extractors (new heads) ---
    medications_node   = extractor_chain(llm, MedicationsOut,    "medications (name, dose, route, frequency, changes, adherence issues, OTC)", "meds")
    allergies_node     = extractor_chain(llm, AllergiesOut,      "allergies and adverse reactions", "allergies")
    vitals_node        = extractor_chain(llm, VitalsOut,         "vitals/measurements mentioned (type, value, unit, when, source)", "vitals")
    history_node       = extractor_chain(llm, HistoryOut,        "past history and comorbidities", "conditions")
    sdoh_node          = extractor_chain(llm, SDOHOut,           "social determinants impacting care", "factors")
    functional_node    = extractor_chain(llm, FunctionalOut,     "functional status (ADLs/IADLs)", "adls")  # parser allows extra fields
    risksafety_node    = extractor_chain(llm, RiskSafetyOut,     "risk and safety factors", "risks")
    education_node     = extractor_chain(llm, EducationOut,      "education topics, teach-back, and barriers", "taught")
    followups_node     = extractor_chain(llm, FollowupsOut,      "follow-ups and tasks (task, who, when_due)", "actions")

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

    # --- composer/map/facts ---
    soap_node     = composer_chain(llm)
    omaha_node    = omaha_chain(llm)
    fact_guard    = factuality_guard_chain(llm)

    g = StateGraph(State)

    # ORIGINAL serial core with validators, then append new heads (each followed by its validator)
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

    # NEW heads (serial; keep order conservative to avoid propagating noise)
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

    # Downstream
    g.add_node("soap",              lambda s: n_soap(s, soap_node))
    g.add_node("omaha",             lambda s: n_omaha(s, omaha_node))
    g.add_node("finalize",          n_finalize)
    g.add_node("final_review",      lambda s: n_final_review(s, fact_guard))

    # SERIAL EDGES (strict order; feel free to reorder later)
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

    # New heads after the core five
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

    # Final composition & scrub
    g.add_edge("v_followups", "soap")
    g.add_edge("soap", "omaha")
    g.add_edge("omaha", "finalize")
    g.add_edge("finalize", "final_review")
    g.add_edge("final_review", END)

    return g.compile(), llm

# ================================
# 6) Runner
# ================================

def summarize_text(transcript: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> str:
    app, llm = build_graph(model=model, temperature=temperature)
    state = app.invoke({"transcript": transcript})
    final_text = state.get("final_text")

    if final_text:
        return final_text

    # Fallback: robust single-pass SOAP + Omaha directly from transcript
    fb = fallback_soap_chain(llm)
    out = fb.invoke({"transcript": transcript})
    return out.content if hasattr(out, "content") else str(out)

# ================================
# 7) CLI
# ================================

def main():
    ap = argparse.ArgumentParser(description="Summarize patient–nurse transcript to SOAP + Omaha (serial, validated, hallucination-scrubbed, extended heads).")
    ap.add_argument("--file", "-f", type=str, help="Path to transcript file. If omitted, read from stdin.")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name (default: gpt-4o-mini)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    ap.add_argument("--keep-greetings", action="store_true", help="Keep greetings at start (default: strip)")
    args = ap.parse_args()

    raw = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    clean = preprocess_text(raw, keep_greetings=args.keep_greetings)
    text = summarize_text(clean, model=args.model, temperature=args.temperature)
    print(text)

if __name__ == "__main__":
    main()
