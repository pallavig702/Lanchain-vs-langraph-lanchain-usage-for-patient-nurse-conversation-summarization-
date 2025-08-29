#!/usr/bin/env python3
"""
agentic_summarizer_cli.py
--------------------------------
Text-only summarization of patient–nurse conversations into:

S: ...
O: ...
A: ...
P: ...
Problems (Omaha): ...
Interventions (Omaha): ...

Pipeline:
- Preprocess: normalize SPEAKER_XX (Role) → Nurse:/Patient:; strip greetings only at start
- LangGraph StateGraph with CoT-wrapped subagents (internal JSON), final output is text only

Usage:
  python agentic_summarizer_cli.py --file path/to/transcript.txt
  cat transcript.txt | python agentic_summarizer_cli.py

Options:
  --model gpt-4o-mini (default), or s/g to what you use (e.g., gpt-4o, gpt-4.1)
  --temperature 0.0 (default)
  --keep-greetings  (optional) keep greetings (disables greeting stripping)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, TypedDict

from pydantic import BaseModel, Field

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangGraph (new API)
from langgraph.graph import StateGraph, START, END

#import os
#os.environ["OPENAI_API_KEY"] = "sk-...yourkey..."
# ================================
# 0) PREPROCESSING (text only)
# ================================

# Normalize "SPEAKER_01 (Nurse): ..." -> "Nurse: ...", "Patient: ..."
SPEAKER_RE = re.compile(r'^\s*SPEAKER_\d+\s*\(([^)]+)\):\s*(.*)$', re.IGNORECASE)

def normalize_transcript(raw: str) -> str:
    """
    Convert diarization lines like 'SPEAKER_01 (Nurse): ...' to a canonical 'Nurse: ...' / 'Patient: ...'.
    Keeps only non-empty lines.
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

# Strict greetings (pure phatic expressions) for the very beginning only.
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

# Cues that clinical talk has started; once seen, stop stripping.
CLINICAL_CUES = [
    re.compile(r"\b(pain|fever|cough|breath|breathing|sleep|dizzi|bp|blood pressure|glucose|sugar|wound|medicat|dose|injec|fall|injur|symptom|since|when|monitor|exercise)\b", re.I),
    re.compile(r"\?$"),  # direct question → often clinical
    re.compile(r"\bdescribe how\b", re.I),
    re.compile(r"\bhow have you been feeling\b", re.I),
]

def strip_greetings_only(normalized_text: str) -> str:
    """
    Remove greeting lines ONLY at the start of the visit. Keep short 'yes/no/okay' lines.
    Stop filtering once a clinical cue appears.
    """
    kept = []
    clinical_started = False

    for line in normalized_text.splitlines():
        if ":" not in line:
            continue
        role, text = line.split(":", 1)
        t = text.strip()

        if clinical_started:
            kept.append(f"{role.strip()}: {t}")
            continue

        # if a clinical cue appears, keep and stop filtering
        if any(p.search(t) for p in CLINICAL_CUES):
            clinical_started = True
            kept.append(f"{role.strip()}: {t}")
            continue

        # keep short yes/no/okay even at start
        if t.lower() in {"yes", "no", "ok", "okay", "alright"}:
            kept.append(f"{role.strip()}: {t}")
            continue

        # drop pure greetings at the very beginning
        if any(p.fullmatch(t) for p in GREETING_PATTERNS):
            continue

        kept.append(f"{role.strip()}: {t}")

    return "\n".join(kept)

def preprocess_text(raw: str, keep_greetings: bool = False) -> str:
    """
    Normalize diarization; optionally strip greetings only at start.
    """
    norm = normalize_transcript(raw)
    return norm if keep_greetings else strip_greetings_only(norm)


# =========================================
# 1) Pydantic Schemas (internal JSON only)
# =========================================

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
    problems: List[str] = Field(default_factory=list)       # Omaha problem labels (short)
    interventions: List[str] = Field(default_factory=list)  # Omaha action labels (e.g., Teaching, Surveillance)


# ================================
# 2) LangGraph State
# ================================

class State(TypedDict, total=False):
    transcript: str
    problems: ProblemsOut
    observations: ObservationsOut
    emo_cog: EmotionCogOut
    interventions: InterventionsOut
    patient_resp: PatientResponseOut
    soap: SOAPOut
    omaha: OmahaOut
    final_text: str
    warnings: List[str]


# =================================
# 3) LLM builder & chain factories
# =================================

def build_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Build a chat model. Requires OPENAI_API_KEY in env.
    """
    return ChatOpenAI(model=model, temperature=temperature, api_key="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"
)

# CoT policy: encourage stepwise thinking but never reveal it
COT_SUFFIX = (
    "Think step by step internally. Do NOT reveal your chain-of-thought. "
    "Only output valid JSON matching the schema."
)

def extractor_chain(llm, schema_model, task_name: str, key_name: str):
    """
    Create a generic extractor chain that returns JSON (Problems, Observations, etc.).
    """
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

def composer_chain(llm):
    """
    Compose a concise SOAP from extractor outputs. Enforces 'no-new-facts'.
    """
    parser = JsonOutputParser(pydantic_object=SOAPOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Compose a concise SOAP note from the provided lists. "
         "Do NOT add any facts not present in the lists. "
         "Do NOT invent vitals/meds/timelines. "
         "Think internally; output JSON for SOAP only."),
        ("user",
         "Problems:\n{problems}\n\n"
         "Observations:\n{observations}\n\n"
         "Emotion/Cognition (verbal):\n{emo_cog}\n\n"
         "Interventions:\n{interventions}\n\n"
         "Patient response:\n{patient_resp}\n\n"
         f"Schema: {parser.get_format_instructions()}\n"
         "Style: 1–2 sentences per S/O/A/P; precise clinical language.")
    ])
    return prompt | llm | parser

def omaha_chain(llm):
    """
    Map to concise Omaha labels (no codes). Problems + actions like Teaching/Guidance/Surveillance.
    """
    parser = JsonOutputParser(pydantic_object=OmahaOut)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Map to concise Omaha labels for problems and interventions. "
         "Examples of actions: Teaching, Guidance, Surveillance, Treatments. "
         "No codes. Think internally; output JSON only."),
        ("user",
            "Problems (bullets):\n{problems}\n\n"
            "Interventions (bullets):\n{interventions}\n\n"
            f"Schema: {parser.get_format_instructions()}\n"
            "Rules: Only select labels clearly justified by the bullets; keep lists short.")
    ])
    return prompt | llm | parser

def fallback_soap_chain(llm):
    """
    Last-resort fallback: produce text-only SOAP+Omaha lines directly from transcript
    if JSON parsing fails somewhere. Still enforces 'no-new-facts'.
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


# ================================
# 4) Nodes (with light resilience)
# ================================

def safe_invoke(chain, payload, empty_value):
    """
    Call a chain and return either parsed output or an empty_value + warning.
    """
    try:
        return chain.invoke(payload), None
    except Exception as e:
        return empty_value, f"{type(e).__name__}: {e}"

def n_problems(state: State, chain):
    out, warn = safe_invoke(chain, {"transcript": state["transcript"]}, ProblemsOut())
    updates = {"problems": out}
    if warn:
        updates.setdefault("warnings", []).append(f"problems_node: {warn}")
    return updates

def n_observations(state: State, chain):
    out, warn = safe_invoke(chain, {"transcript": state["transcript"]}, ObservationsOut())
    updates = {"observations": out}
    if warn:
        updates.setdefault("warnings", []).append(f"observations_node: {warn}")
    return updates

def n_emo_cog(state: State, chain):
    out, warn = safe_invoke(chain, {"transcript": state["transcript"]}, EmotionCogOut())
    updates = {"emo_cog": out}
    if warn:
        updates.setdefault("warnings", []).append(f"emo_cog_node: {warn}")
    return updates

def n_interventions(state: State, chain):
    out, warn = safe_invoke(chain, {"transcript": state["transcript"]}, InterventionsOut())
    updates = {"interventions": out}
    if warn:
        updates.setdefault("warnings", []).append(f"interventions_node: {warn}")
    return updates

def n_patient_resp(state: State, chain):
    out, warn = safe_invoke(chain, {"transcript": state["transcript"]}, PatientResponseOut())
    updates = {"patient_resp": out}
    if warn:
        updates.setdefault("warnings", []).append(f"patient_resp_node: {warn}")
    return updates

def n_soap(state: State, chain):
    out, warn = safe_invoke(chain, {
        "problems": state.get("problems", ProblemsOut()).problems,
        "observations": state.get("observations", ObservationsOut()).observations,
        "emo_cog": state.get("emo_cog", EmotionCogOut()).emotion_cognition,
        "interventions": state.get("interventions", InterventionsOut()).interventions,
        "patient_resp": state.get("patient_resp", PatientResponseOut()).patient_response,
    }, SOAPOut(S="", O="", A="", P=""))
    updates = {"soap": out}
    if warn:
        updates.setdefault("warnings", []).append(f"soap_node: {warn}")
    return updates

def n_omaha(state: State, chain):
    out, warn = safe_invoke(chain, {
        "problems": state.get("problems", ProblemsOut()).problems,
        "interventions": state.get("interventions", InterventionsOut()).interventions,
    }, OmahaOut())
    updates = {"omaha": out}
    if warn:
        updates.setdefault("warnings", []).append(f"omaha_node: {warn}")
    return updates

def n_finalize(state: State) -> State:
    """
    Return the final human-readable text. If SOAP is empty (parse failures), we’ll fill later via fallback.
    """
    if "soap" in state and isinstance(state["soap"], SOAPOut) and state["soap"].S:
        soap = state["soap"]
        omaha = state.get("omaha", OmahaOut())
        final_text = (
            f"S: {soap.S}\n"
            f"O: {soap.O}\n"
            f"A: {soap.A}\n"
            f"P: {soap.P}\n"
            f"Problems (Omaha): {', '.join(omaha.problems) if omaha.problems else 'None'}\n"
            f"Interventions (Omaha): {', '.join(omaha.interventions) if omaha.interventions else 'None'}"
        )
        return {"final_text": final_text}
    # If SOAP missing, finalization will be handled by caller (fallback path).
    return {}


# ================================
# 5) Graph builder + runner
# ================================

def build_graph(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Build the StateGraph using the current LangGraph API.
    """
    llm = build_llm(model=model, temperature=temperature)

    problems_node = extractor_chain(llm, ProblemsOut, "patient-reported problems/symptoms", "problems")
    observations_node = extractor_chain(llm, ObservationsOut, "objective observations", "observations")
    emo_cog_node = extractor_chain(llm, EmotionCogOut, "verbalized emotions/cognitive states", "emotion_cognition")
    interventions_node = extractor_chain(llm, InterventionsOut, "nurse interventions", "interventions")
    patient_resp_node = extractor_chain(llm, PatientResponseOut, "patient response", "patient_response")
    soap_node = composer_chain(llm)
    omaha_node = omaha_chain(llm)

    g = StateGraph(State)
    g.add_node("problems", lambda s: n_problems(s, problems_node))
    g.add_node("observations", lambda s: n_observations(s, observations_node))
    g.add_node("emo_cog", lambda s: n_emo_cog(s, emo_cog_node))
    g.add_node("interventions", lambda s: n_interventions(s, interventions_node))
    g.add_node("patient_resp", lambda s: n_patient_resp(s, patient_resp_node))
    g.add_node("soap", lambda s: n_soap(s, soap_node))
    g.add_node("omaha", lambda s: n_omaha(s, omaha_node))
    g.add_node("finalize", n_finalize)

    g.add_edge(START, "problems")
    g.add_edge("problems", "observations")
    g.add_edge("observations", "emo_cog")
    g.add_edge("emo_cog", "interventions")
    g.add_edge("interventions", "patient_resp")
    g.add_edge("patient_resp", "soap")
    g.add_edge("soap", "omaha")
    g.add_edge("omaha", "finalize")
    g.add_edge("finalize", END)

    return g.compile(), llm  # return llm too (for fallback)


def summarize_text(transcript: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> str:
    """
    Run the full graph. If internal JSON parsing fails (rare), use a fallback text-only chain.
    """
    app, llm = build_graph(model=model, temperature=temperature)
    state = app.invoke({"transcript": transcript})
    '''
    final_text = state.get("final_text")
    '''
    #####
    text = state.get("final_text")
    final_text = review_with_transcript(text, transcript)
    print("Not Reviewed Output:\n", text)
    print("Reviewed Output:\n", final_text)
    ######
    if final_text:
        return final_text

    # Fallback path (e.g., if SOAP parsing failed and finalize had nothing to print)
    fb = fallback_soap_chain(llm)
    out = fb.invoke({"transcript": transcript})
    # .content should be already formatted S/O/A/P + Omaha lines
    return out.content if hasattr(out, "content") else str(out)

from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

def review_with_transcript(generated_summary: str, transcript: str) -> str:
    """
    Review agent that checks the generated SOAP/Omaha output against transcript.
    Removes or flags unsupported information.
    """
    review_prompt = f"""
    You are a strict reviewer. Compare the GENERATED SUMMARY with the TRANSCRIPT.
    - Keep only facts present in the transcript.
    - Remove anything not supported by the transcript.
    - Ensure problems and interventions are only based on transcript evidence.

    TRANSCRIPT:
    {transcript}

    GENERATED SUMMARY:
    {generated_summary}

    Return the cleaned summary.
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm([
        SystemMessage(content="You are a factuality reviewer for clinical transcripts."),
        HumanMessage(content=review_prompt)
    ])
    return response.content


# ================================
# 6) CLI
# ================================

def main():
    ap = argparse.ArgumentParser(description="Summarize patient–nurse transcript to SOAP + Omaha (text only).")
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

    # Preprocess (normalize; strip greetings unless override)
    clean = preprocess_text(raw, keep_greetings=args.keep_greetings)

    # Summarize
    text = summarize_text(clean, model=args.model, temperature=args.temperature)
    print(text)
    
    

if __name__ == "__main__":
    main()
