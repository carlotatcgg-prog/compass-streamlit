import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

import requests
import streamlit as st
from datetime import datetime


# DEBUG (temporal)





# -----------------------------
# Config
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

BUNDLE_PROMPTS = {
    "landing_email": "Generate a landing page + email funnel bundle.",
    "ads_pack": "Generate an ads pack with TikTok hooks, Meta ads, and Google keywords.",
    "offer_pricing": "Generate offer and pricing page copy.",
    "outreach_pack": "Generate an outreach pack with cold emails and LinkedIn DMs.",
    "custom": "Generate custom deliverables based on the user's specific request.",
}

# -----------------------------
# Validation gate (binary valid/invalid)
# -----------------------------
class Deliverable(TypedDict):
    title: str
    content: str

class ValidDeliverables(TypedDict):
    systemSummary: str
    primaryDeliverable: Deliverable
    supportingDeliverable: Deliverable
    executionChecklist: List[str]
    nextActions: List[str]

def create_fallback(raw_text: str) -> ValidDeliverables:
    return {
        "systemSummary": "Draft system generated. You can regenerate to refine.",
        "primaryDeliverable": {
            "title": "Generated Draft",
            "content": raw_text or "No content was generated. Please try again.",
        },
        "supportingDeliverable": {
            "title": "Notes",
            "content": "This draft was auto-generated due to formatting issues.",
        },
        "executionChecklist": ["Review the draft", "Regenerate the system if needed", "Customize before use"],
        "nextActions": ["Regenerate system"],
    }

def validate_deliverables(parsed: Any, raw_text: str) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return {"valid": False, "data": create_fallback(raw_text), "rawText": raw_text}

    required = ["systemSummary", "primaryDeliverable", "supportingDeliverable", "executionChecklist", "nextActions"]
    for k in required:
        if k not in parsed:
            return {"valid": False, "data": create_fallback(raw_text), "rawText": raw_text}

    if not isinstance(parsed["systemSummary"], str) or not parsed["systemSummary"].strip():
        return {"valid": False, "data": create_fallback(raw_text), "rawText": raw_text}

    pd = parsed["primaryDeliverable"]
    sd = parsed["supportingDeliverable"]

    if not (isinstance(pd, dict) and isinstance(pd.get("title"), str) and isinstance(pd.get("content"), str) and pd["content"].strip()):
        return {"valid": False, "data": create_fallback(raw_text), "rawText": raw_text}
    if not (isinstance(sd, dict) and isinstance(sd.get("title"), str) and isinstance(sd.get("content"), str) and sd["content"].strip()):
        return {"valid": False, "data": create_fallback(raw_text), "rawText": raw_text}

    ec = parsed["executionChecklist"]
    na = parsed["nextActions"]

    if not (isinstance(ec, list) and len(ec) > 0 and all(isinstance(x, str) for x in ec)):
        return {"valid": False, "data": create_fallback(raw_text), "rawText": raw_text}
    if not (isinstance(na, list) and len(na) > 0 and all(isinstance(x, str) for x in na)):
        return {"valid": False, "data": create_fallback(raw_text), "rawText": raw_text}

    data: ValidDeliverables = {
        "systemSummary": parsed["systemSummary"].strip(),
        "primaryDeliverable": {"title": pd["title"].strip(), "content": pd["content"].strip()},
        "supportingDeliverable": {"title": sd["title"].strip(), "content": sd["content"].strip()},
        "executionChecklist": [x.strip() for x in ec if isinstance(x, str) and x.strip()],
        "nextActions": [x.strip() for x in na if isinstance(x, str) and x.strip()],
    }
    return {"valid": True, "data": data, "rawText": raw_text}

def safe_extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None

    clean = text.strip()

    # Remove fenced blocks if any
    clean = re.sub(r"^```json\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^```\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean, flags=re.IGNORECASE).strip()

    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean, flags=re.IGNORECASE)
    if m:
        clean = m.group(1).strip()

    first = clean.find("{")
    last = clean.rfind("}")
    if first != -1 and last != -1 and last > first:
        clean = clean[first:last + 1]

    try:
        return json.loads(clean)
    except Exception:
        return None

def call_gemini(prompt: str, business_type: str, bundle: str, project_title: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return {
            "ok": False,
            "error": "Service temporarily unavailable. Please try again in a moment.",
            "rawText": "",
        }
    bundle_instructions = BUNDLE_PROMPTS.get(bundle, BUNDLE_PROMPTS["custom"])

    system_prompt = f"""
You are a B2B marketing and product operator with 10+ years of experience launching products and writing conversion copy.

RULES:
- Ask zero follow-up questions. Make reasonable assumptions.
- Write in clear, modern English. No fluff, no motivational language, no corporate jargon.
- Be direct and actionable. Every sentence should serve a purpose.
- If the user prompt is vague, choose a sensible default business context.
- Output ONLY valid JSON. No markdown, no code blocks, no explanations outside the JSON.
- Never return plain text or markdown.
- Never omit any fields.
- If unsure, make reasonable assumptions.
- If content quality is low, still return the structure.

You MUST return ONLY valid JSON matching this EXACT schema:
{{
  "systemSummary": "string - A concise 2-3 sentence overview of the business system",
  "primaryDeliverable": {{
    "title": "string - Name of the primary deliverable (e.g., 'Landing Page Copy')",
    "content": "string - The full content of the primary deliverable"
  }},
  "supportingDeliverable": {{
    "title": "string - Name of the supporting deliverable (e.g., 'Email Sequence')",
    "content": "string - The full content of the supporting deliverable"
  }},
  "executionChecklist": [
    "string - Action item 1",
    "string - Action item 2",
    "string - Action item 3"
  ],
  "nextActions": [
    "string - Immediate next step 1",
    "string - Immediate next step 2"
  ]
}}

{bundle_instructions}

Write like you're advising a founder who needs to ship this week. Be specific, not generic.
""".strip()

    user_prompt = f"""Business Type: {business_type}
Bundle: {bundle}
Project Title: {project_title}
Request: {prompt}

Generate the complete deliverables package as JSON.
""".strip()

    url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": system_prompt + "\n\n" + user_prompt}
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
        },
    }

    try:
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code != 200:
            return {
                "ok": False,
                "error": f"Gemini error {r.status_code}: {r.text[:500]}",
                "rawText": f"Gemini failed with status {r.status_code}: {r.text}",
            }

        data = r.json()

        # Typical Gemini response path:
        # candidates[0].content.parts[0].text
        text = ""
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            text = json.dumps(data)

        raw_text = text if isinstance(text, str) else json.dumps(text)
        parsed = safe_extract_json(raw_text)
        return {"ok": True, "rawText": raw_text, "parsed": parsed}
    except Exception as e:
        return {"ok": False, "error": f"Request failed: {e}", "rawText": f"Error: {e}"}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Compass", page_icon="🧭", layout="centered")

st.title("🧭 Compass")
st.caption("Generate structured systems and deliverables. Always returns a draft fallback (never empty).")
# -----------------------------
# Monetization / Validation
# -----------------------------
FREE_LIMIT = 3

if "uses_left" not in st.session_state:
    st.session_state.uses_left = FREE_LIMIT

st.info(f"Free generations left: {st.session_state.uses_left} / {FREE_LIMIT}")

with st.sidebar:
    st.subheader("Inputs")
    business_type = st.text_input("Business type", value="general")
    bundle = st.selectbox("Bundle", options=list(BUNDLE_PROMPTS.keys()), index=4)  # custom
    project_title = st.text_input("Project title", value="Compass Generation")
    show_raw = st.toggle("Show raw output (debug)", value=False)

prompt = st.text_area("Prompt", height=160, placeholder="Describe what you want Compass to generate...")

generate = st.button("Generate", type="primary", use_container_width=True)


if generate:
    if st.session_state.uses_left <= 0:
        st.warning("Free limit reached.")
        st.stop()

    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()

    with st.spinner("Generating…"):
        ai = call_gemini(
            prompt.strip(),
            business_type.strip() or "general",
            bundle,
            project_title.strip() or "Compass Generation",
        )

    # ⬇️ RESTA AQUÍ (justo después de generar)
    st.session_state.uses_left -= 1

    raw_text = ai.get("rawText", "")
    if not ai.get("ok"):
        fallback = create_fallback(raw_text or ai.get("error", "Unknown error"))
        st.warning("Generated Draft (fallback)")
        st.subheader("System Summary")
        st.write(fallback["systemSummary"])
        st.subheader(fallback["primaryDeliverable"]["title"])
        st.text(fallback["primaryDeliverable"]["content"])
        st.subheader(fallback["supportingDeliverable"]["title"])
        st.text(fallback["supportingDeliverable"]["content"])
        st.subheader("Execution Checklist")
        for item in fallback["executionChecklist"]:
            st.write(f"- {item}")
        st.subheader("Next Actions")
        for item in fallback["nextActions"]:
            st.write(f"- {item}")

        st.error(ai.get("error", "Unknown error"))
        if show_raw:
            with st.expander("Raw output"):
                st.code(raw_text or "(empty)", language="text")
        st.stop()

    validated = validate_deliverables(ai.get("parsed"), raw_text)
    valid = validated["valid"]
    data = validated["data"]

    if not valid:
        st.warning("Generated Draft (fallback)")

    st.subheader("System Summary")
    st.write(data["systemSummary"])

    st.subheader(data["primaryDeliverable"]["title"])
    st.text(data["primaryDeliverable"]["content"])

    st.subheader(data["supportingDeliverable"]["title"])
    st.text(data["supportingDeliverable"]["content"])

    st.subheader("Execution Checklist")
    for item in data["executionChecklist"]:
        st.write(f"- {item}")

    st.subheader("Next Actions")
    for item in data["nextActions"]:
        st.write(f"- {item}")

    if show_raw:
        with st.expander("Raw output"):
            st.code(raw_text or "(empty)", language="text")

if st.session_state.uses_left <= 0:
    st.markdown("---")
    st.subheader("🚀 Unlock Compass")

    st.write(
        "Compass builds **systems**, not just text.\n\n"
        "Get unlimited generations, saved projects, and advanced bundles."
    )

    email = st.text_input("Get early access (email)")

    if st.button("Join early access"):
        if email and "@" in email:
            with open("early_access.csv", "a") as f:
                f.write(f"{email},{datetime.utcnow().isoformat()}\n")
            st.success("You're on the list! We'll be in touch.")
        else:
            st.error("Please enter a valid email.")

