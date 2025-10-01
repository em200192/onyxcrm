import streamlit as st
from PIL import Image
import os
import io
import fitz  # PyMuPDF
import json
import re
from pathlib import Path
import base64
import torch
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# >>> NEW: std libs for GCS export
import csv
from io import StringIO
from datetime import datetime, timezone
from uuid import uuid4

# --- CONFIGURATION ---
load_dotenv()
st.set_page_config(
    page_title="المساعد الذكي",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS & DIRECTORIES ---

# --- NEW: Dual KBs (GL user guide + Errors) ---
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploaded_files"
CACHE_DIR = BASE_DIR / "cache"
IMAGE_CACHE_DIR = CACHE_DIR / "solution_images"
GUIDE_PDF_DEFAULT = str((BASE_DIR / "data" / "user_guides" / "Gl.pdf").resolve())
ERRORS_PDF_DEFAULT = str((BASE_DIR / "uploaded_files" / "erros and solution.pdf").resolve())

GUIDE_KB_PATH = CACHE_DIR / "gl_guide_kb.json"
ERRORS_KB_PATH = CACHE_DIR / "gl_errors_kb.json"
GUIDE_IMG_DIR = CACHE_DIR / "guide_images"
os.makedirs(GUIDE_IMG_DIR, exist_ok=True)






from dataclasses import dataclass

# --- Arabic normalization ---
# --- Arabic normalization & helpers ---
AR_NORM_MAP = {'أ':'ا','إ':'ا','آ':'ا','ى':'ي','ة':'ه','ؤ':'و','ئ':'ي','ٰ':'','ـ':''}
AR_DIACRITICS = r'[\u064B-\u065F\u0670]'

def ar_norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(AR_DIACRITICS, '', s)
    return ''.join(AR_NORM_MAP.get(ch, ch) for ch in s)

def contains(text: str, *words) -> bool:
    t = ar_norm(text)
    return any(ar_norm(w) in t for w in words)

# Domain synonym buckets (normalized forms)
SYN = {
    "create": {"اضاف","اضافه","انشاء","ادخال","تسجيل","تعريف"},
    "report": {"تقرير","تقارير"},
    "request": {"طلب","طلبات"},
    "voucher_word": {"سند","سندات"},
    "payment": {"صرف","سندسرف","سندالصرف","سنداتالصرف"},
    "receipt": {"قبض","سندقبض","سنداتالقبض"},
    "currency": {"عمله","عملة","صرف عمله","صرف عملة","شراء عملة","بيع عملة"},
    "supplier": {"مورد","موردين"},
    "customer": {"عميل","عملاء"},
    "bank": {"بنك","مصرف","الحساب البنكي","حساب بنكي","حساب بنك"},
    "cash_fund": {"صندوق","الخزنة","خزنة","cash fund"},
    "checkbook": {"دفتر شيكات","دفتر الشيكات","شيك","شيكات","checkbook"}
}


def has_any(text: str, terms: set[str]) -> bool:
    t = ar_norm(text)
    return any(ar_norm(w) in t for w in terms)




@dataclass
class Slots:
    doc_type: str|None = None      # "screen" | "report" | None
    voucher_type: str|None = None  # "receipt" | "payment" | None
    object_key: str|None = None    # "bank" | "cash_fund" | "checkbook" | ...
    action: str|None = None        # "create" | None

def extract_slots(query: str) -> dict:
    q = ar_norm(query)
    slots = {
        "doc_type": None,          # "screen" | "report"
        "voucher_family": None,    # "payment" | "receipt"
        "party": None,             # "supplier" | "customer"
        "wants_request": False,    # user said "طلب"
        "wants_voucher": False,    # user said "سند"
        "money_exchange": False,   # mentions عملة
        "action_create": False,    # add/create
    }

    if contains(q, *SYN["report"]):  slots["doc_type"] = "report"
    if contains(q, *SYN["create"]):  slots["action_create"] = True

    if contains(q, *SYN["voucher_word"]): slots["wants_voucher"] = True
    if contains(q, *SYN["request"]):      slots["wants_request"] = True

    if contains(q, *SYN["payment"]): slots["voucher_family"] = "payment"
    if contains(q, *SYN["receipt"]): slots["voucher_family"] = slots["voucher_family"] or "receipt"

    if contains(q, *SYN["currency"]): slots["money_exchange"] = True

    if contains(q, *SYN["supplier"]): slots["party"] = "supplier"
    if contains(q, *SYN["customer"]): slots["party"] = slots["party"] or "customer"

    # If user asked "كيف/اضافة" with no "تقرير", prefer screens
    if slots["doc_type"] is None and (slots["action_create"] or "كيف" in q):
        slots["doc_type"] = "screen"

    return slots




KB_PATH = CACHE_DIR / "knowledge_base.json"
CONFIDENCE_THRESHOLD = 0.55
SUGGESTION_THRESHOLD = 0.40
MAX_FOLLOW_UP_ATTEMPTS = 2


# >>> NEW: GCS config (optional)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")              # e.g. "crm-escalations"
GCS_CREDENTIALS_JSON = os.getenv("GCS_CREDENTIALS_JSON","")    # optional path to SA JSON

# Place near other constants
ACTIVE_INTENT_THRESHOLDS = {
    "close_to_current": 0.40,   # similar to current issue
    "new_issue_candidate": 0.45,# similar to some other KB entry
    "margin": 0.08              # current similarity must exceed next-best by this margin
}

# --- Escalation intent: quick patterns (Arabic + EN) ---
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

# --- CORE AI & KNOWLEDGE BASE FUNCTIONS ---

# --- Escalation intent: quick patterns (Arabic + EN) ---
ESCALATE_PATTERNS = [
    "وصلني بالمستشار", "وصلني بمستشار", "وصلني بالدعم", "اتواصل مع الدعم",
    "التواصل مع الدعم", "اريد الدعم", "اريد اتواصل مع فريق الدعم", "كلم الدعم",
    "اكلم الدعم", "مستشار", "موظف", "بشري", "بريد الدعم", "رقم الدعم",
    "اتصال بالدعم", "حولني للدعم", "اريد شخص", "اريد مسؤول", "اريد موظف",
    "talk to human", "talk to agent", "human support", "connect me to support",
    "transfer to agent", "escalate", "escalation", "support team", "helpdesk"
]

# --- SMART FOLLOW-UP CONSTANTS ---
EXPLICIT_ESCALATE_PATTERNS = [
    "مستشار", "الدعم", "support", "help desk", "اريد اتواصل", "اريد التواصل",
    "اتصلوا بي", "كلمني", "ابغى اكلم", "تواصل مع الفريق", "اريد احد يتواصل معي"
]
NEGATIVE_CUES = [
    "لم أستطع","ما قدرت","لا أستطيع","محجوب","لا تظهر الشاشة","رفض","خطأ",
    "نفس المشكلة","لم يعمل","لا يعمل","ما نفع","ما زالت","لا يزال"
]


# --- Fuzzy helpers for safer suggestions ---



# --- NEW: source router for initial queries ---
ERROR_LIKE_PAT = re.compile(
    r"(?:\bmsg\b|\berror\b|\bmessage\b|رسالة|مشكلة)\s*(?:no|رقم)?\s*\d{2,}|"
    r"(?:\b\d{3,}\b\s*(?:msg|error|message))|"
    r"(?:الرسالة\s*رقم\s*\d{2,})",
    re.IGNORECASE
)

# Add this pattern (3+ digits with only whitespace around)
DIGITS_ONLY_PAT = re.compile(r"^\s*\d{3,}\s*$")

# Accept Arabic or English plain 3+ digits
DIGITS_ONLY_PAT = re.compile(r"^\s*(?:\d|[٠-٩]){3,}\s*$")



@st.cache_resource
def create_competition_router_chain():
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""
أنت محكِّم مصادر. لديك سؤال للمستخدم و"أفضل المرشحين" من مصدرين:
- Errors: مشاكل فعلية برسائل وأرقام وحلول
- Guide: شاشات/خطوات/شروحات

المدخلات:
- السؤال: {query}
- مرشحو الأخطاء (id | score | message_number | message_text | reason):
{errors_block}

- مرشحو الدليل (id | score | title | category | snippet):
{guide_block}

قرِّر المصدر الأفضل لحل السؤال الآن، اعتمادًا على النصوص ذاتها لا على الكلمات المفتاحية.
أجب JSON فقط:
{{
  "source": "errors" | "guide",
  "confidence": 0.0_to_1.0,
  "reason": "<ملخص قصير>"
}}
""",
        input_variables=["query", "errors_block", "guide_block"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)


def _topk_indices(scores: torch.Tensor, k: int):
    k = min(k, scores.shape[-1])
    vals, idxs = torch.topk(scores, k)
    return idxs.tolist(), vals.tolist()

def _build_errors_block(kb, idxs, vals):
    lines = []
    for i, s in zip(idxs, vals):
        it = kb[i]
        num = str(it.get("message_number","")).strip()
        msg = (it.get("message_text") or "").replace("\n"," ")[:140]
        rsn = (it.get("reason") or "").replace("\n"," ")[:120]
        lines.append(f"{i} | {s:.3f} | {num} | {msg} | {rsn}")
    return "\n".join(lines)

def _build_guide_block(kb, idxs, vals):
    lines = []
    for i, s in zip(idxs, vals):
        it = kb[i]
        ttl = (it.get("title") or "").replace("\n"," ")[:140]
        cat = (it.get("category") or "").strip()
        snp = (it.get("body") or "").replace("\n"," ")[:160]
        lines.append(f"{i} | {s:.3f} | {ttl} | {cat} | {snp}")
    return "\n".join(lines)


# ---- Smart error-id extraction ----
ERROR_CONTEXT_WORDS = {
    "رساله","رسالة","خطا","خطأ","كود","رمز","رقم المشكلة","رقم الرسالة","الرسالة",
    "message","msg","error","code","problem","issue"
}
CURRENCY_WORDS = {"sar","ريال","درهم","usd","egp","جنيه","eur","$","٪","%","ر.س","د.إ"}
DATE_PAT = re.compile(r"\b(?:19|20)\d{2}\b")   # 1900–2099 (year-like)
TIME_PAT = re.compile(r"\b\d{1,2}[:٫.]\d{2}\b")# e.g., 12:30 or 12٫30

@st.cache_resource
def create_error_id_extractor_chain():
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""
استخرج أرقام رسائل الأخطاء المقصودة من نص المستخدم (إن وُجدت).
- تجاهل الأسعار والمبالغ والأوقات والتواريخ وأرقام الهواتف.
- اعتبر الكلمات: رسالة، خطأ، كود، رمز، رقم المشكلة، message, error, code, msg, problem, issue مؤشرات قوية.
أجب JSON فقط:
{"error_ids": ["..."], "confidence": 0.0_to_1.0}

النص:
{text}
""",
        input_variables=["text"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

def extract_error_ids_smart(text: str) -> dict:
    s = to_english_digits((text or "").strip())
    s_low = s.lower()

    # candidates: 3–6 digits (covers 3, 4, 5, 6)
    matches = list(re.finditer(r"(?<!\d)(\d{3,6})(?!\d)", s))
    if not matches:
        return {"ids": [], "confidence": 0.0}

    valid = st.session_state.get("valid_message_numbers", set()) or set()

    def is_noise(m):
        n = m.group(1)
        win = s_low[max(0, m.start()-15): m.end()+15]
        if TIME_PAT.search(win): return True
        if DATE_PAT.search(win) and len(n) == 4: return True
        if any(w in win for w in CURRENCY_WORDS): return True
        # simple phone-ish: 7–10+ digits in full input (we're only scoring 3–6 but keep guard)
        if re.search(r"\b\d{7,}\b", s): return True
        return False

    scored = []
    for m in matches:
        n = m.group(1)
        if is_noise(m):
            continue
        score = 0.35
        # known KB number = strongest
        if n in valid:
            score = 1.0
        # context words near number
        ctx = s_low[max(0, m.start()-12): m.end()+12]
        if any(w in ctx for w in ERROR_CONTEXT_WORDS):
            score += 0.4
        # bare-number query (just the number) is very strong
        if s.strip() == n:
            score += 0.5
        scored.append((n, min(score, 1.0)))

    scored.sort(key=lambda x: x[1], reverse=True)
    ids = [n for n, _ in scored]
    conf = scored[0][1] if scored else 0.0

    # LLM tie-breaker if still unsure
    if conf < 0.6:
        try:
            raw = create_error_id_extractor_chain().invoke({"text": text})["text"]
            data = _json_only(raw) or {}
            llm_ids = [to_english_digits(str(x)) for x in (data.get("error_ids") or [])]
            # merge (preserve order/uniques)
            ids = list(dict.fromkeys(llm_ids + ids))
            conf = max(conf, float(data.get("confidence", 0.0)))
        except Exception:
            pass

    return {"ids": ids, "confidence": conf}




def route_by_competition(query: str, k: int = 6, margin: float = 0.03) -> dict:
    """
    Pure retrieval competition:
    1) Embed query
    2) Take top-k from errors + top-k from guide
    3) If best score differs by a clear margin -> pick that source
    4) Else ask LLM arbiter (no keyword rules)
    Returns: {"source": "errors"|"guide", "confidence": float,
              "best_error_idx": int|None, "best_guide_idx": int|None}
    """
    _ensure_kb_matrices()

    q_norm = to_english_digits((query or "").strip())
    mentioned_nums = re.findall(r"\b\d{3,}\b", q_norm)
    valid_nums = st.session_state.get("valid_message_numbers", set()) or set()

    if mentioned_nums:
        # If *any* mentioned number exists in the errors KB → force errors
        if any(n in valid_nums for n in mentioned_nums):
            return {"source": "errors", "confidence": 0.99, "best_error_idx": None, "best_guide_idx": None}
        # If user gave a number but it's not in KB, still route to errors
        # so your existing logic can escalate gracefully.
        return {"source": "errors", "confidence": 0.75, "best_error_idx": None, "best_guide_idx": None}

    errors_kb = st.session_state.get("gl_errors_kb") or []
    guide_kb  = st.session_state.get("gl_guide_kb") or []
    if not errors_kb and not guide_kb:
        return {"source":"guide","confidence":0.0,"best_error_idx":None,"best_guide_idx":None}

    model = get_embedding_model()
    qemb = model.encode((query or "").strip(), convert_to_tensor=True)

    # scores
    best_err = best_gui = None
    err_idxs = err_vals = gui_idxs = gui_vals = []
    if errors_kb and "gl_errors_matrix" in st.session_state:
        s_err = util.cos_sim(qemb, st.session_state.gl_errors_matrix)[0]
        err_idxs, err_vals = _topk_indices(s_err, k)
        best_err = (err_idxs[0], float(err_vals[0])) if err_idxs else (None, 0.0)
    else:
        best_err = (None, 0.0)

    if guide_kb and "gl_guide_matrix" in st.session_state:
        s_gui = util.cos_sim(qemb, st.session_state.gl_guide_matrix)[0]
        gui_idxs, gui_vals = _topk_indices(s_gui, k)
        best_gui = (gui_idxs[0], float(gui_vals[0])) if gui_idxs else (None, 0.0)
    else:
        best_gui = (None, 0.0)

    # 1) clear winner by margin
    be_i, be_s = best_err
    bg_i, bg_s = best_gui
    if be_i is not None and (be_s >= bg_s + margin):
        return {"source":"errors","confidence":min(1.0, be_s), "best_error_idx":be_i, "best_guide_idx":bg_i}
    if bg_i is not None and (bg_s >= be_s + margin):
        return {"source":"guide","confidence":min(1.0, bg_s), "best_error_idx":be_i, "best_guide_idx":bg_i}

    # 2) too close → ask LLM arbiter on the actual top-k texts
    arb = create_competition_router_chain()
    errors_block = _build_errors_block(errors_kb, err_idxs, err_vals) if err_idxs else "(no candidates)"
    guide_block  = _build_guide_block (guide_kb,  gui_idxs,  gui_vals)  if gui_idxs  else "(no candidates)"
    try:
        raw = arb.invoke({
            "query": query,
            "errors_block": errors_block,
            "guide_block": guide_block
        })["text"]
        data = _json_only(raw) or {}
        src = data.get("source", "guide")
        conf = float(data.get("confidence", 0.55))
        return {"source": src, "confidence": conf, "best_error_idx": be_i, "best_guide_idx": bg_i}
    except Exception:
        # fallback to simple tie-break: pick the higher raw score even if close
        if (be_s >= bg_s):  # deterministic
            return {"source":"errors","confidence":min(1.0, be_s), "best_error_idx":be_i, "best_guide_idx":bg_i}
        else:
            return {"source":"guide","confidence":min(1.0, bg_s), "best_error_idx":be_i, "best_guide_idx":bg_i}

# --- Fallback JSON extractor (use if you don't already have one) ---
def _json_only(s: str) -> dict:
    m = re.search(r'\{.*\}', s, re.DOTALL)
    return json.loads(m.group(0)) if m else {}



@st.cache_resource
def create_tip_generation_chain():
    """
    Creates an LLM chain that reads a block of text and generates a single,
    engaging 'Did you know?' tip from it.
    """
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""You are a helpful assistant who creates 'Did you know?' tips from technical ERP software manuals.
Read the following topic content. Your task is to find or generate one single, interesting, and useful tip for a user.

- The tip should be a complete sentence and written in engaging Arabic.
- It should be a practical piece of advice, a shortcut, or an interesting fact.
- Do NOT simply repeat a "Usage" or "Purpose" sentence. Find a real insight.
- If you cannot find a good, interesting tip in the text, you MUST respond with an empty JSON object.

TOPIC CONTENT:
---
{content}
---

Respond with a valid JSON object only, like this:
{{"tip": "<your generated tip>"}}
""",
        input_variables=["content"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

@st.cache_resource
def create_followup_router_chain():
    """
    Decides if a follow-up question is about the error solution itself
    or a general "how-to" query that requires searching the user guide.
    """
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""You are an intent classifier. A user is currently solving an error. Your task is to classify their follow-up question.

The error's solution involves these steps: {solution_context}

The user's follow-up question is: "{query}"

Does the user's question ask for more detail about the specific error's solution, or is it a more general "how-to" question about a feature (like a screen or a process) mentioned in the solution?

Choose one of the following intents:
- "troubleshoot_error": If the user is asking directly about the error or its solution steps.
- "search_guide": If the user is asking a general "how-to" question (e.g., "how do I open that screen?", "what is a ...?").

Respond with a valid JSON object only, like this:
{{"intent": "<troubleshoot_error|search_guide>"}}
""",
        input_variables=["solution_context", "query"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)


# --- LLM source router: "errors" vs "guide" ---
@st.cache_resource
def create_source_router_chain():
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""
صنّف رسالة المستخدم إلى مصدر واحد:
- "errors" إذا كانت مشكلة برسالة/كود، خصوصًا مع رقم 3+ أرقام.
- "guide" لأسئلة الشاشات/الخطوات/التقارير (كيف/إضافة/تعريف/تقرير).

النص: {q}

JSON فقط:
{{"source":"errors"|"guide"}}
""",
        input_variables=["q"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)



def decide_source_smart(q: str) -> str:
    raw = (q or "").strip()

    # (A) Smart number-first detector (3–6 digits, Arabic/English, context-aware)
    det = extract_error_ids_smart(raw)
    if det["ids"] and det["confidence"] >= 0.6:
        st.session_state["__detected_error_ids__"] = det["ids"]
        return "errors"

    # (B) Strong how-to/report cues → guide
    if any(c in raw for c in ["كيف","طريقة","اضاف","إضافة","تعريف","شاشة","تقرير","report"]):
        return "guide"

    # (C) Fallback: quick patterns
    raw_norm = to_english_digits(raw.lower())
    if re.search(r"(رسالة|خط[اأ]ء|problem|error|code|message|msg)\s*\d{3,6}", raw_norm):
        return "errors"

    # (D) LLM tie-breaker
    try:
        r = create_source_router_chain().invoke({"q": q})["text"]
        return (_json_only(r) or {}).get("source", "guide")
    except Exception:
        return "guide"

from difflib import SequenceMatcher

AR_LETTERS = r"\u0600-\u06FF"

def _arabic_english_normalize(s: str) -> str:
    s = to_english_digits((s or "").lower())
    # keep Arabic letters, ASCII letters/digits, spaces
    s = re.sub(fr"[^{AR_LETTERS}a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _lexical_ratio(a: str, b: str) -> float:
    """Simple, fast similarity (0..1) based on SequenceMatcher."""
    return SequenceMatcher(None, _arabic_english_normalize(a), _arabic_english_normalize(b)).ratio()

def _token_coverage(query: str, target: str) -> float:
    qt = _arabic_english_normalize(query).split()
    tt = set(_arabic_english_normalize(target).split())
    if not qt:
        return 0.0
    hit = sum(1 for w in qt if w in tt)
    return hit / len(qt)

def _explicit_escalation_requested(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in EXPLICIT_ESCALATE_PATTERNS)

def _stuck_score(user_text: str, last_assistant_reply: str) -> int:
    """Tiny heuristic: +1 if user expresses failure, +1 if assistant repeated itself,
    +1 if no new action words appear."""
    score = 0
    u = (user_text or "")
    a = (last_assistant_reply or "")
    if any(c in u for c in NEGATIVE_CUES): score += 1
    # repeated advice (same reply twice)
    if st.session_state.get("last_assistant_hash") == hash(a.strip()):
        score += 1
    # little novelty: no verbs suggesting an action
    if not re.search(r"(احفظ|افتح|اذهب|تأكد|فعّل|أعد|حدّث|جرّب|غيّر|أغلق|أعد فتح)", a):
        score += 1
    return score




def get_did_you_know_tip(user_query: str | None = None) -> dict | None:
    """
    Returns: {"text": tip, "title": title, "page": page}
    - If user_query is provided: pick the most similar tip (embeddings).
    - Else: rotate a daily random tip (stable per day).
    """
    tips = load_gl_dyk_index()
    if not tips:
        return None

    # Context-aware mode
    if user_query:
        try:
            qemb = get_embedding_model().encode(user_query.strip(), convert_to_tensor=True)
            mat = torch.tensor([t["embedding"] for t in tips])
            sims = util.cos_sim(qemb, mat)[0]
            idx = int(torch.argmax(sims).item())
            best = tips[idx]
            return {"text": best["tip"], "title": best["title"], "page": best["page"]}
        except Exception:
            pass

    # Daily-rotating deterministic pick
    seed = int(datetime.utcnow().strftime("%Y%m%d"))
    idx = seed % len(tips)
    t = tips[idx]
    return {"text": t["tip"], "title": t["title"], "page": t["page"]}



def render_did_you_know_sidebar():
    if build_gl_dyk_index():  # ensures ready
        tip = get_did_you_know_tip(None)
        if tip:
            with st.sidebar:
                st.markdown("### ℹ️ هل تعلم؟ / Did you know?")
                st.info("هل تعلم انك تستطيع معرفة مصروفات وإيرادات الشركة من خلال قائمة الدخل")



# --- NEW: Did-You-Know (DYK) index for GL guide ---
DYK_INDEX_PATH = CACHE_DIR / "gl_guide_dyk.json"


@st.cache_resource
def build_gl_dyk_index(force: bool = False) -> bool:
    """
    Upgraded version: Uses an LLM to generate high-quality, engaging tips
    from each section of the GL guide KB.
    """
    try:
        # We add force=True here to ensure we regenerate with the new logic
        if (not force) and DYK_INDEX_PATH.exists() and not force:
            # If you want to force a rebuild every time, remove this check
            return True

        # Ensure the main guide KB is loaded
        if "gl_guide_kb" not in st.session_state or not st.session_state.gl_guide_kb:
            if not load_gl_guide_kb():
                print("Could not load GL Guide KB to build DYK index.")
                return False

        guide = st.session_state.get("gl_guide_kb") or []
        tip_chain = create_tip_generation_chain()
        all_tips = []

        print("🧠 Starting AI-powered 'Did You Know?' tip generation...")
        with st.spinner("Generating smart tips with AI..."):
            for i, sec in enumerate(guide):
                title = (sec.get("title") or "").strip()
                page = int(sec.get("page") or 0)
                body = (sec.get("body") or "")

                # Skip if there's no body or the page number is invalid
                if not body or page == 0:
                    continue

                print(f"  -> Analyzing topic {i + 1}/{len(guide)}: {title}")
                try:
                    # Ask the LLM to generate a tip from the body text
                    response_text = tip_chain.invoke({"content": body})["text"]
                    tip_data = _json_only(response_text)

                    if tip_data and "tip" in tip_data and tip_data["tip"]:
                        all_tips.append({
                            "tip": tip_data["tip"],
                            "title": title[:160],
                            "page": page,
                        })
                except Exception as e:
                    # This can happen if the LLM returns an empty response or fails
                    print(f"    -> No tip generated for '{title}'. Skipping.")
                    continue

        if not all_tips:
            print("No tips were generated by the AI.")
            return False

        # Embed the new, high-quality tips
        print("Embedding new tips...")
        model = get_embedding_model()
        embs = model.encode([c["tip"] for c in all_tips], convert_to_tensor=True)
        for i, c in enumerate(all_tips):
            c["embedding"] = embs[i].tolist()

        with open(DYK_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(all_tips, f, ensure_ascii=False, indent=2)
        print(f"Successfully generated and saved {len(all_tips)} smart tips.")
        return True
    except Exception as e:
        st.warning(f"An error occurred while building the 'Did You Know?' index: {e}")
        return False

def load_gl_dyk_index() -> list:
    if DYK_INDEX_PATH.exists():
        try:
            with open(DYK_INDEX_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []




def generate_guide_instructions(section: dict, user_query: str = "") -> dict:
    """
    Turns a GL guide section into a clean answer with a format
    based on the section's category.
    Returns {"answer": str, "images": [paths...]}.
    """
    llm = get_llm_model()
    title = (section.get("title") or "").strip()
    body  = (section.get("body")  or "").strip()
    category = section.get("category", "Inputs") # Default to 'Inputs' if category is missing

    # --- PROMPT SELECTION LOGIC ---
    prompt = ""

    # 1. Format for System Inputs & Operations (Your existing instructed format)
    if category in ["Inputs", "Operations"]:
        prompt = f"""
أنت كاتب أدلة استخدام محترف. صِغ إجابة عملية موجّهة للمستخدم بناءً على قسم من دليل مستخدم Onyx ERP (GL).

سؤال المستخدم: "{user_query}"

القواعد:
- اتبع هذا القالب بدقة.
- لخّص وأعد الصياغة لتصبح خطوات قابلة للتنفيذ.

### 📘 {title}
**🧭 الهدف:** سطر واحد يوضح ما الذي سينجزه المستخدم.
### 🛠️ طريقة الاستخدام
(حوّل النص إلى قائمة خطوات مرقمة وواضحة: 1. ... 2. ... 3. ...)
### 💡 ملاحظات هامة
(لخّص أي نقاط مهمة هنا، إن وجدت)

نص الدليل لتحويله:
\"\"\"{body[:3500]}\"\"\"
""".strip()

    # 2. Format for System Configuration (New "System Usage Configuration" format)
    elif category == "Configuration":
        prompt = f"""
أنت خبير في نظام Onyx ERP. اشرح إعداد التكوين التالي بطريقة واضحة وموجزة.

سؤال المستخدم: "{user_query}"
موضوع التكوين: "{title}"

القواعد:
- اتبع هذا القالب بدقة.
- ركز على "لماذا" و "متى" يستخدم هذا الإعداد.

### ⚙️ تهيئة استخدام النظام: {title}
**الغرض من الإعداد:**
(اشرح هنا بوضوح الغرض من هذا الإعداد وتأثيره على النظام في فقرة واحدة.)

**خيارات الإعداد:**
(لخّص الخيارات المتاحة لهذا الإعداد وماذا يعني كل خيار في نقاط.)

نص الدليل للملخص:
\"\"\"{body[:3500]}\"\"\"
""".strip()

    # 3. Format for System Reports (New "Purpose of Use" format)
    elif category == "Reports":
        prompt = f"""
أنت محلل أعمال تشرح فائدة تقرير مالي من نظام Onyx ERP.

سؤال المستخدم: "{user_query}"
اسم التقرير: "{title}"

القواعد:
- اتبع هذا القالب بدقة.
- ركز على الفائدة العملية للتقرير.

### 📊 الغرض من استخدام التقرير: {title}
**ما هو هذا التقرير؟**
(اشرح بوضوح ما هي البيانات التي يعرضها هذا التقرير في فقرة واحدة.)

**لماذا هو مهم؟**
(وضح ما هي القرارات التي يساعد هذا التقرير الإدارة على اتخاذها، ومن هم المستخدمون الرئيسيون له.)

نص الدليل للملخص:
\"\"\"{body[:3500]}\"\"\"
""".strip()

    # --- LLM INVOCATION AND RESPONSE ---
    try:
        resp = llm.invoke(prompt)
        md = (resp.content or "").strip()
    except Exception:
        # Fallback to a simple format if the LLM fails
        md = f"### 📘 {title}\n\n{body}"

    imgs = section.get("images") or []
    return {"answer": md, "images": imgs}

@st.cache_resource
def create_escalation_judge_chain():
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""
أنت "مُقيِّم متابعة". لديك:
- المشكلة: رقم {message_number} | نص: {message_text}
- السبب: {reason}
- الحل الكامل: {solution}
- آخر ردين للمساعد: [{a1}] ثم [{a2}]
- آخر ردين للمستخدم: [{u1}] ثم [{u2}]
- رسالة المستخدم الآن: {user_text}
- الإجراء المقترح من المساعد الآن: {assistant_proposal}

قيِّم:
1) هل هذا الإجراء **جديد ومفيد** (وليس تكرارًا واضحًا)؟
2) هل المستخدم **محجوب/مقيّد** (صلاحيات/شاشة لا تفتح/جرّب ولم ينجح)؟
3) احتمالية أننا بحاجة للتصعيد الآن.

أجب JSON فقط:
{{
  "new_action_quality": 0.0_to_1.0,
  "blocked_signals": ["..."],
  "escalate_prob": 0.0_to_1.0,
  "short_reason": "<سبب مختصر>"
}}
""",
        input_variables=[
            "message_number","message_text","reason","solution",
            "a1","a2","u1","u2","user_text","assistant_proposal"
        ]
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

@st.cache_resource
def create_escalation_intent_chain():
    llm = get_llm_model()
    prompt = """
أنت مصنِّف نوايا. لديك آخر رسالتين للمستخدم (إن وجِدتا) ورسالة المستخدم الحالية.
حدّد هل يطلب المستخدم صراحةً التواصل مع "مستشار/دعم بشري" أو يريد "تصعيد" الآن.

أجب بصيغة JSON فقط:
{"escalate": true | false, "confidence": 0.0}

- آخر رسائل المستخدم: [1] {last_user_1}  |  [2] {last_user_2}
- الرسالة الحالية: {user_text}
"""
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=prompt,
            input_variables=["last_user_1", "last_user_2", "user_text"]
        ),
        verbose=False
    )

def _wants_handoff(user_text: str) -> bool:
    """Generic, dynamic check: (1) fast patterns, (2) LLM intent."""
    t = (user_text or "").strip().lower()
    if any(p in t for p in ESCALATE_PATTERNS):
        return True
    try:
        msgs = _last_msgs(2)  # you already have this helper
        chain = create_escalation_intent_chain()
        raw = chain.invoke({
            "last_user_1": msgs["u1"],
            "last_user_2": msgs["u2"],
            "user_text": user_text
        })["text"]
        data = _json_from_model_text(raw)
        return bool(data.get("escalate")) and float(data.get("confidence", 0.0)) >= 0.5
    except Exception:
        return False

@st.cache_resource
def create_blocked_detector_chain():
    llm = get_llm_model()
    prompt = """
أنت مصنِّف لحالة التعثّر/الانسداد لدى المستخدم.
إذا كانت الرسالة الحالية تدل أن المستخدم **غير قادر على المتابعة** (لا أستطيع/لا أصل/لا أملك صلاحية/جرّبت ولم ينجح/...),
فأجب بأن المستخدم "blocked".

JSON فقط:
{"blocked": true | false, "confidence": 0.0}

- آخر رسائل المستخدم: [1] {last_user_1}  |  [2] {last_user_2}
- الرسالة الحالية: {user_text}
"""
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=prompt,
            input_variables=["last_user_1", "last_user_2", "user_text"]
        ),
        verbose=False
    )

@st.cache_resource
def get_llm_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("لم يتم العثور على مفتاح Google API. يرجى إضافته إلى ملف .env")
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.1)

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def _last_assistant_message_text() -> str:
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "assistant":
            return msg.get("content", "")[:800]
    return ""

def _numbers_catalog(max_chars: int = 6000) -> str:
    nums = sorted(list(st.session_state.get("valid_message_numbers", set())))
    s = ", ".join(nums)
    return s[:max_chars]

def _json_from_model_text(text: str):
    # robust JSON extraction (code fence tolerant)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))

@st.cache_resource
@st.cache_resource
def create_intent_router_chain():
    llm = get_llm_model()
    router_prompt = """
أنت وكيل تصنيف ذكي لدعم نظام Onyx ERP. لديك مشكلة نشطة حاليًا وربما يذكر المستخدم رقم رسالة مختلف،
وربما يطلب فتح مشكلة/تذكرة جديدة.

مهمتك: تحديد نية رسالة المستخدم التالية بدقة، وإرجاع JSON فقط وفق المخطط المحدد.

السياق:
- رقم المشكلة النشطة: {current_number}
- نص المشكلة النشطة (مختصر): {current_text}
- السبب (مختصر): {current_reason}
- الحل (مختصر): {current_solution}
- آخر رد من المساعد: {last_assistant_msg}
- أرقام الرسائل الصحيحة الموجودة في الدليل: [{valid_numbers}]

المخرجات المطلوبة (JSON فقط):
{{
  "intent": "follow_up" | "new_issue" | "resolved" | "open_new" | "other",
  "mentioned_numbers": [<strings>],
  "step_number": <int or null>,
  "new_issue_number": <string or null>,
  "confidence": <float 0..1>,
  "reason": "<مختصر يبرر القرار>"
}}

قواعد اتخاذ القرار:
1) إذا كانت الرسالة تعني أن المشكلة انحلّت (مثل: "تم حلها" / "اشتغلت") → intent="resolved".
2) إذا ذكرت الرسالة رقم رسالة موجود في الدليل ومختلف عن {current_number} → intent="new_issue" واملأ new_issue_number.
3) إذا طلب المستخدم فتح مشكلة/تذكرة/قضية جديدة (مثل: "افتح مشكلة جديدة" / "افتح تذكرة" / "open new issue")
   بدون تحديد رقم معروف → intent="open_new".
4) إذا كانت الرسالة متابعة للمشكلة الحالية (سؤال/توضيح/ذكر خطوة) → intent="follow_up" واستخراج step_number إن وُجد.
5) إن لم ينطبق شيء مما سبق → "other".

أمثلة سريعة:
- "افتح مشكلة جديدة" → open_new.
- "open new ticket" → open_new.
- "669" أو "٦٦٩" → new_issue with new_issue_number="669".
- "رقم خمسة" (بعد سؤال عن الخطوة) → follow_up, step_number=5.
- "تم حلها" → resolved.

رسالة المستخدم:
\"\"\"{user_text}\"\"\"\n
أجب بالـ JSON فقط.
"""
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=router_prompt,
            input_variables=[
                "current_number","current_text","current_reason","current_solution",
                "last_assistant_msg","valid_numbers","user_text"
            ]
        ),
        verbose=False
    )

def _is_negative_reply(t: str) -> bool:
    s = (t or "").strip().lower()
    # Arabic & EN negatives / polite declines
    negatives = [
        "لا", "لا شكرا", "لا شكراً", "خلاص", "انتهى", "تمام شكرا", "شكراً", "ما اريد", "ما احتاج",
        "مو لازم", "بلا", "no", "nope", "nah", "no thanks", "no thank you", "all good", "im good", "i'm good"
    ]
    return any(s == n or s.startswith(n + " ") for n in negatives)

def _asks_for_human(t: str) -> bool:
    s = (t or "").strip().lower()
    # phrases that clearly mean "connect me to human support"
    triggers = [
        "مستشار", "ادعم", "الدعم الفني", "الفني", "اتواصل", "تواصل", "اريد التواصل", "اريد اتواصل",
        "حوّلني", "حولني", "كلم", "كلّم", "support", "human", "agent", "advisor", "escalate"
    ]
    return any(k in s for k in triggers)

def _last_msgs(n: int = 2):
    users, assistants = [], []
    for m in reversed(st.session_state.chat_history):
        if m["role"] == "assistant":
            assistants.append(m.get("content", ""))
        elif m["role"] == "user":
            users.append(m.get("content", ""))
        if len(users) >= n and len(assistants) >= n:
            break
    while len(users) < n: users.append("")
    while len(assistants) < n: assistants.append("")
    return {
        "u1": users[0], "u2": users[1] if len(users) > 1 else "",
        "a1": assistants[0], "a2": assistants[1] if len(assistants) > 1 else "",
    }


def _is_permission_related(issue: dict, user_text: str) -> bool:
    """Return True if the active issue or the user's last message looks like a permissions problem."""
    parts = [
        (issue or {}).get("message_text") or "",
        (issue or {}).get("reason") or "",
        (issue or {}).get("solution") or "",
        user_text or "",
    ]
    hay = " ".join(parts).lower()

    # Arabic + English cues
    perm_kws = [
        "صلاحيات", "صلاحية", "تفويض", "غير مخول", "غير مصرح", "الدور", "دور المستخدم",
        "permission", "permissions", "authorization", "authorisation", "access denied", "role"
    ]
    return any(k in hay for k in perm_kws)


# ---------- LLM selectors (no embeddings) ----------
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def _json_only(s: str) -> dict:
    m = re.search(r'\{.*\}', s, re.DOTALL)
    return json.loads(m.group(0)) if m else {}

@st.cache_resource
def create_llm_title_selector_chain():
    """
    Round 1: given ALL section titles (id, title, category),
    pick up to k ids that best match the user's intent.
    """
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""
أنت مُنتقٍ ذكي لأقسام دليل Onyx ERP. مهمتك اختيار أفضل الأقسام بالاعتماد على العناوين فقط.

مبادئ عامة:
- إذا كان سؤال المستخدم يتضمن "كيف/إضافة/إنشاء/تعريف" ففضّل الشاشات والإجراءات (وليس التقارير).
- إذا ذُكرت "قبض" ففضّل "سند القبض" وتجنّب "سند الصرف" و"تقارير سند القبض".
- إذا ذُكرت "صرف" ففضّل "سند الصرف" وتجنّب "صرف عملة" و"تقارير سند الصرف" ما لم يُذكر "عملة".
- إذا ذُكرت "عملة" ففضّل موضوعات "صرف عملة/طلب صرف عملة".
- "مورد" يلمح غالبًا إلى سند صرف، "عميل" يلمح غالبًا إلى سند قبض.
- إذا طلب المستخدم "تقرير" فاختر تقارير، وإلا ففضّل الشاشات التنفيذية.

سؤال/هدف المستخدم:
{query}

العناصر (id | title | category):
{items}

أرجع JSON فقط بهذا الشكل:
{{
  "ids": [<up to {k} integers>],
  "short_reason": "<لماذا هذه الاختيارات>"
}}
""",
        input_variables=["query", "items", "k"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

@st.cache_resource
def create_llm_candidate_ranker_chain():
    """
    Round 2: given small candidate set with snippets, rank to top 3.
    """
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""
أنت مُعيد ترتيب دقيق. لديك مرشّحات صغيرة من دليل Onyx ERP.
رتّبها من الأكثر مناسبة لسؤال المستخدم إلى الأقل، واحذر من الخلط بين "سند" و"طلب" و"تقرير".

سؤال المستخدم:
{query}

المرشحات (id | title | category | snippet):
{cands}

أجب JSON فقط:
{{
  "ranked_ids": [<أفضل 3 ids بالترتيب>],
  "confidence": 0.0_to_1.0,
  "short_reason": "<لماذا>"
}}
""",
        input_variables=["query", "cands"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

def is_follow_up_on_current_issue(user_text: str) -> bool:
    issue = st.session_state.get("current_issue")
    if not (st.session_state.get("solution_provided") and issue):
        return False
    text = (user_text or "").strip()
    msg_no = str(issue.get("message_number", "")).strip()
    if msg_no and re.search(rf'\b{re.escape(msg_no)}\b', text):
        return True
    followup_cues = [
        "لم أستطع","مش واضح","غير واضح","كيف","أشرح","ما المقصود","لم يعمل",
        "ما زال","لا يزال","أعد","رجاء","المزيد","لم يتم","أين أجد","خطوة",
        "بعد ذلك","ثم","ماذا أفعل","المشكلة السابقة","هذه الخطوة","لم أتمكن"
    ]
    if any(p in text for p in followup_cues):
        nums = re.findall(r'\d+', text)
        if not nums or (msg_no and all(n == msg_no for n in nums)):
            return True
    try:
        embedding_model = get_embedding_model()
        query_emb = embedding_model.encode(text, convert_to_tensor=True)
        ref_text = " ".join([
            issue.get("message_text","") or "",
            issue.get("reason","") or "",
            issue.get("solution","") or ""
        ]).strip()
        ref_emb = torch.tensor(issue.get("embedding")) if issue.get("embedding") else embedding_model.encode(ref_text, convert_to_tensor=True)
        sim = util.cos_sim(query_emb, ref_emb).item()
        return sim >= 0.35
    except Exception:
        return False

def llm_parse_page_as_json(page_image_bytes):
    llm = get_llm_model()
    prompt = """
    أنت خبير في استخراج البيانات. حلل صورة الصفحة التالية من دليل تقني.
    لكل قسم خطأ تجده، استخرج المعلومات التالية:
    - message_number
    - message_text
    - location
    - reason
    - solution
    - note (إن وجدت)
    أرجع قائمة JSON فقط. إن لم يوجد شيء فأرجع [].
    """
    encoded_image = base64.b64encode(page_image_bytes).decode('utf-8')
    message = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {
        "url": f"data:image/png;base64,{encoded_image}"}}])
    try:
        response = llm.invoke([message])
        json_text = re.search(r'\[.*\]', response.content, re.DOTALL).group(0)
        return json.loads(json_text)
    except Exception as e:
        st.warning(f"AI parsing failed for a page, skipping. Error: {e}")
        return []

def llm_extract_message_number_from_image(image_bytes):
    llm = get_llm_model()
    prompt = """
    أنت خبير في التعرف على الأرقام.
    انظر إلى صورة رسالة الخطأ هذه. استخرج الرقم الموجود بجوار "Message No -".
    إذا وجدت الرقم، أرجعه فقط كعدد صحيح. إذا لم تتمكن من العثور عليه، أرجع "null".
    لا تقم بتضمين أي نص إضافي، فقط الرقم أو "null".
    """
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    message = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {
        "url": f"data:image/png;base64,{encoded_image}"}}])
    try:
        response = llm.invoke([message])
        cleaned_response = response.content.strip()
        if cleaned_response.isdigit():
            return cleaned_response
        return None
    except Exception as e:
        st.error(f"Error extracting message number from image: {e}")
        return None

def generate_instructional_response(result):
    llm = get_llm_model()
    context = f"""
    - Error Message Text: {result.get('message_text')}
    - Reason: {result.get('reason')}
    - Solution: {result.get('solution')}
    - Note: {result.get('note', 'N/A')}
    """
    prompt_template = """
    أنت وكيل دعم فني خبير ومساعد متخصص في نظام "Onyx ERP". مهمتك هي تحويل بيانات الخطأ التالية إلى دليل إرشادي واضح للمستخدم.
    اتبع هذه القواعد بدقة:
    1. خاطب المستخدم مباشرة بأسلوب ودود ومهني.
    2. استخدم تنسيق Markdown التالي: `### 🔍 ما الذي يحدث؟` و `### 🛠️ كيفية الإصلاح`.
    3. تحت "ما الذي يحدث؟"، اشرح سبب الخطأ بعبارات بسيطة.
    4. تحت "كيفية الإصلاح"، قم بتحويل خطوات الحل إلى قائمة مرقمة وواضحة.
    5. إذا كانت هناك "ملاحظة"، أضفها في النهاية تحت عنوان **"💡 ملاحظة هامة:"**.
    **السياق:**
    {context}
    الآن، قم بإنشاء الدليل الإرشادي للمستخدم باللغة العربية.
    """
    final_prompt = prompt_template.format(context=context)
    response = llm.invoke(final_prompt)
    image_path = result.get('image_path')
    return {"answer": response.content, "images": [image_path] if image_path else []}

def classify_intent(user_text: str) -> dict:
    text = (user_text or "").strip().lower()
    resolved_cues = ["تم حلها", "انحلت", "اشتغلت", "thank you", "solved", "works now"]
    if any(cue in text for cue in resolved_cues):
        return {"intent": "resolved", "confidence": 0.95}
    issue = st.session_state.get("current_issue") or {}
    curr_no = str(issue.get("message_number", "")).strip()
    nums = re.findall(r'\d+', text)
    if nums and (not curr_no or any(n != curr_no for n in nums)):
        return {"intent": "new_issue", "confidence": 0.85}
    try:
        llm = get_llm_model()
        context = {
            "current_message_number": curr_no,
            "current_message_text": (issue.get("message_text") or "")[:1000],
            "current_reason": (issue.get("reason") or "")[:800],
            "current_solution": (issue.get("solution") or "")[:1200],
        }
        prompt = f"""
أنت مصنف نوايا رسائل محادثة لدعم فني لنظام Onyx ERP.
لدينا مشكلة نشطة برقم: {context["current_message_number"] or "غير معروف"}.

صنّف الرسالة التالية إلى واحدة فقط من القيم:
- follow_up
- new_issue
- resolved
- other

المعطيات عن المشكلة الحالية (اختصار):
[النص]: {context["current_message_text"]}
[السبب]: {context["current_reason"]}
[الحل]: {context["current_solution"]}

نص المستخدم: \"\"\"{user_text}\"\"\"

أجب بصيغة JSON فقط:
{{"intent": "<follow_up|new_issue|resolved|other>", "confidence": 0.0}}
        """.strip()
        resp = llm.invoke(prompt)
        m = re.search(r'\{.*\}', resp.content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and "intent" in data:
                return {"intent": data.get("intent"), "confidence": float(data.get("confidence", 0.6))}
    except Exception:
        pass
    if len(text) <= 60 or any(k in text for k in ["كيف","ما","مش واضح","لم أستطع","لم يعمل","خطوة"]):
        return {"intent": "follow_up", "confidence": 0.6}
    return {"intent": "other", "confidence": 0.5}



@st.cache_resource
def create_guide_reranker_chain():
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""
أنت مصنف يختار أفضل قسم من دليل مستخدم Onyx ERP (GL) ليتوافق مع نية المستخدم.

المدخلات:
- سؤال/هدف المستخدم: {query}
- المرشّحات (أقصى 6). لكل مرشّح: id, title, snippet (من أول نص الصفحة)

اختر أفضل مرشّح واحد فقط، أو "none" إن لم يكن أي مرشّح مناسباً بوضوح.

أجب JSON فقط:
{{
  "best_id": "<id أو none>",
  "confidence": 0.0,
  "short_reason": "<سبب مختصر>"
}}

المرشّحات:
{candidates}
""",
        input_variables=["query", "candidates"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

@st.cache_resource
def create_troubleshooting_chain():
    llm = get_llm_model()
    prompt_template = """
أنت مساعد دعم لنظام "Onyx ERP" وتعالج المشكلة النشطة فقط (لا تفتح قضايا جديدة).
قدّم إرشادًا قصيرًا خطوة بخطوة بسؤال تحقق واحد في كل رسالة.

المشكلة:
- رقم الرسالة: {message_number}
- النص: {message_text}
- السبب: {reason}
- الحل المقترح: {solution}

مبادئ ذكاء عامة عند الفشل في خطوة:
- تحقق من الصلاحيات/الدور والفرع والشركة.
- تعارض إعدادات: خيار يُعطل خيارًا آخر.
- ضرورة الحفظ وإعادة فتح الشاشة/تحديث الجلسة/مسح الكاش.
- بطء/انقطاع الاتصال أو قفل السجل.
- سياسات الجودة/الاعتماد تمنع التفعيل.

سجل المحادثة: {chat_history}
رسالة المستخدم: {question}

إن شعرت أن الخطوات استُنفدت، اكتب فقط: "نحتاج تصعيد".
رد موجز مباشر:
"""
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=prompt_template,
            input_variables=["message_number","message_text","reason","solution","chat_history","question"]
        ),
        verbose=False
    )





def build_gl_guide_kb(pdf_path: str) -> bool:
    """
    Final version: Correctly handles image extraction for topics that start
    on the same page where a previous topic ends, preventing image mismatch.
    """
    try:
        from sentence_transformers import SentenceTransformer
        _HAS_EMB = True
        print("SentenceTransformer library found. Embeddings will be generated.")
    except ImportError:
        _HAS_EMB = False
        print("Warning: SentenceTransformer not found. Run 'pip install sentence-transformers' to generate embeddings.")

    # --- Configuration ---
    base_dir = Path(__file__).parent
    cache_dir = base_dir / "cache"
    guide_kb_path = cache_dir / "gl_guide_kb.json"
    guide_img_dir = cache_dir / "guide_images"
    os.makedirs(guide_img_dir, exist_ok=True)

    # --- Helper Functions ---
    def clean_body_text(s: str) -> str:
        s = s or ""
        s = re.sub(r'solution_image_start:.*?solution_image_end:', '', s, flags=re.DOTALL | re.IGNORECASE)
        s = re.sub(r'\s+', ' ', s.replace("\r", "")).strip()
        return s

    def save_image(doc, xref, out_dir: Path, topic_title: str, img_idx: int):
        # Clean the title to use in the filename
        temp_title = re.sub(r'[^a-zA-Z0-9\s]', '', topic_title)
        safe_title = re.sub(r'\s+', '_', temp_title).lower().strip('_')[:30]
        if not safe_title:
            safe_title = "guide_topic"  
        
        try:
            img = doc.extract_image(xref)
            ext = img.get("ext", "png").lower()
            # Use a more descriptive filename
            # Inside the save_image function in build_gl_guide_kb
            out_path = out_dir / f"{safe_title}_img_{img_idx}.{ext}"
            with open(out_path, "wb") as f:
                f.write(img["image"])
            return str(out_path)
        except Exception:
            return None

    def get_marker_pos(doc, page_range, marker_text, after_pos=None):
        # Finds the first occurrence of a marker AFTER a specific starting position.
        for page_num in page_range:
            # Skip pages before the starting position's page
            if after_pos and page_num < after_pos["page"]:
                continue

            page = doc[page_num]
            rects = page.search_for(marker_text, quads=False)
            for rect in rects:
                # If on the same page, ensure the marker is vertically below the start
                if after_pos and page_num == after_pos["page"] and rect.y0 < after_pos["y"]:
                    continue
                # This is the first valid marker found
                return {"page": page_num, "y": float(rect.y0)}
        return None

    def collect_images_for_topic(doc, page_range, topic_start_pos, topic_title):
        images_found = []
        # **MODIFICATION**: The search for the start marker now begins AFTER the topic_title
        start_marker = get_marker_pos(doc, page_range, "solution_image_start:", after_pos=topic_start_pos)
        if not start_marker:
            return []

        end_marker = get_marker_pos(doc, page_range, "solution_image_end:", after_pos=start_marker)
        if not end_marker:
            end_marker = {"page": page_range.stop - 1, "y": 9999}

        start_page, end_page = start_marker["page"], end_marker["page"]
        save_idx = 0

        for p_num in range(start_page, end_page + 1):
            page = doc[p_num]
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                img_bbox = page.get_image_bbox(img_info)

                is_after_start = (p_num > start_page) or (p_num == start_page and img_bbox.y1 > start_marker["y"])
                is_before_end = (p_num < end_page) or (p_num == end_page and img_bbox.y0 < end_marker["y"])

                if is_after_start and is_before_end:
                    saved_path = save_image(doc, xref, guide_img_dir, topic_title, save_idx)
                    if saved_path:
                        images_found.append(saved_path)
                        save_idx += 1
        return images_found

    # --- Main Execution ---
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error: Could not open PDF file '{pdf_path}': {e}")
        return False

    topic_re = re.compile(
        r"(?P<ar_title>.*?)\s*topic_title:\s*(?P<eng_title>.*?)\n"
        r"category:\s*(?P<category>.*?)\n"
        r"answer_start:\s*(?P<answer_content>.*?)\s*answer_end:",
        re.DOTALL | re.IGNORECASE
    )

    full_text = "".join(page.get_text().replace("\r", "") + "\n" for page in doc)

    char_count = 0
    page_map = []
    for i, page in enumerate(doc):
        text_len = len(page.get_text().replace("\r", "")) + 1
        page_map.append({'start': char_count, 'end': char_count + text_len, 'page_num': i})
        char_count += text_len

    def get_pos_for_char(char_index):
        page_num = next((p['page_num'] for p in page_map if p['start'] <= char_index < p['end']), 0)
        page_start_char = page_map[page_num]['start']
        # This is an approximation for the Y coordinate, but it's sufficient for ordering
        y_coord_approx = (char_index - page_start_char)
        return {"page": page_num, "y": y_coord_approx}

    def get_pages_for_match(match):
        start_char, end_char = match.span()
        start_page = get_pos_for_char(start_char)["page"]
        end_page = get_pos_for_char(end_char)["page"]
        return range(start_page, end_page + 1)

    matches = list(topic_re.finditer(full_text))
    if not matches:
        print("Extraction failed: Could not find any topics matching the specified structure.")
        return False

    print(f"Found {len(matches)} topics. Extracting content...")

    sections = []
    for m in matches:
        ar_title = m.group("ar_title").strip()
        eng_title = m.group("eng_title").strip().replace('(', '( ').replace(')', ' )')  # Clean up parens
        eng_title = re.sub(r'\s+', ' ', eng_title)

        full_title = f"{ar_title} / {eng_title}" if ar_title and eng_title else ar_title or eng_title


        print(f" - Processing: {full_title}")

        page_range = get_pages_for_match(m)

        # Get the precise starting position of this topic's title
        topic_start_pos = get_marker_pos(doc, page_range, "topic_title:")

        # Pass this starting position to the image collector
        images = collect_images_for_topic(doc, page_range, topic_start_pos, full_title)

        raw_answer_content = m.group("answer_content")

        # keep old: ar_title, eng_title already parsed by your regex
        full_title = f"{ar_title} / {eng_title}" if ar_title and eng_title else ar_title or eng_title

        title_norm_ar = ar_norm(ar_title)
        title_norm_en = (eng_title or "").strip().lower()

        def infer_doc_type():
            # Any obvious "تقارير" -> report, else screen (procedure)
            if "تقارير" in ar_title or "reports" in title_norm_en:
                return "report"
            return "screen"

        def infer_voucher_type():
            if has_any(ar_title, SYN["receipt"]): return "receipt"
            if has_any(ar_title, SYN["payment"]): return "payment"
            return None

        def infer_object_key():
            # map from the title (ar/en) without hardcoding every page
            if has_any(ar_title, SYN["bank"]) or "bank" in title_norm_en:
                return "bank"
            if has_any(ar_title, SYN["cash_fund"]) or "cash fund" in title_norm_en or "cash funds" in title_norm_en:
                return "cash_fund"
            if has_any(ar_title, SYN["checkbook"]) or "checkbook" in title_norm_en:
                return "checkbook"
            if "receipt voucher" in title_norm_en:
                return "receipt_voucher"
            if "payment voucher" in title_norm_en:
                return "payment_voucher"
            # fallback None; not every screen is a simple object
            return None

        sections.append({
            "title": full_title,
            "ar_title": ar_title,
            "en_title": eng_title,
            "category": m.group("category").strip(),
            "body": clean_body_text(raw_answer_content),
            "images": images,
            # NEW lightweight ontology fields
            "doc_type": infer_doc_type(),
            "voucher_type": infer_voucher_type(),
            "object_key": infer_object_key(),
        })

    # --- Generate Embeddings ---
    if _HAS_EMB and sections:
        try:
            print("\nGenerating embeddings...")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            texts_to_embed = [f"Title: {s['title']}\nCategory: {s['category']}\nContent: {s['body']}" for s in sections]
            embeddings = model.encode(texts_to_embed, show_progress_bar=True)
            for i, section in enumerate(sections):
                section["embedding"] = embeddings[i].tolist()
            print("Embeddings generated successfully.")
        except Exception as e:
            print(f"Error during embedding generation: {e}")

    with open(guide_kb_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully built the GL Guide KB with {len(sections)} topics.")
    return True

def load_gl_guide_kb() -> bool:
    if GUIDE_KB_PATH.exists():
        try:
            with open(GUIDE_KB_PATH, "r", encoding="utf-8") as f:
                st.session_state.gl_guide_kb = json.load(f)
            return True
        except Exception:
            return False
    return False



def build_errors_kb(pdf_path: str) -> bool:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        st.error(f"تعذر فتح ملف الأخطاء والحلول: {e}")
        return False

    all_items = []
    for i, page in enumerate(doc):
        # Extract last image on the page as the "Solution Image" (if any)
        solution_image_path = None
        images_on_page = page.get_images(full=True)
        if images_on_page:
            try:
                base_image = doc.extract_image(images_on_page[-1][0])
                # reuse IMAGE_CACHE_DIR just like your old pipeline
                out_path = IMAGE_CACHE_DIR / f"errors_p{i+1}_solution.png"
                with open(out_path, "wb") as f:
                    f.write(base_image["image"])
                solution_image_path = str(out_path)
            except Exception:
                pass

        # Parse structured error data from page
        try:
            pix = page.get_pixmap(dpi=150)
            page_image_bytes = pix.tobytes("png")
            parsed = llm_parse_page_as_json(page_image_bytes)
        except Exception:
            parsed = []

        # Attach image_path to each parsed item
        for it in parsed:
            if solution_image_path:
                it["image_path"] = solution_image_path
            all_items.append(it)

    if not all_items:
        st.warning("لم أعثر على رسائل خطأ منظمة في ملف الأخطاء والحلول.")
        return False

    embedding_model = get_embedding_model()
    texts = [f"{p.get('message_text','')} {p.get('reason','')}" for p in all_items]
    embs = embedding_model.encode(texts, convert_to_tensor=True)
    for i, p in enumerate(all_items):
        p["embedding"] = embs[i].tolist()

    with open(ERRORS_KB_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    # refresh valid numbers for router/safety
    st.session_state.valid_message_numbers = {
        str(p.get('message_number','')).strip()
        for p in all_items if str(p.get('message_number','')).strip()
    }
    return True

def load_errors_kb() -> bool:
    if ERRORS_KB_PATH.exists():
        try:
            with open(ERRORS_KB_PATH, "r", encoding="utf-8") as f:
                st.session_state.gl_errors_kb = json.load(f)
            st.session_state.valid_message_numbers = {
                str(p.get('message_number','')).strip()
                for p in st.session_state.gl_errors_kb if str(p.get('message_number','')).strip()
            }
            return True
        except Exception:
            return False
    return False


def load_knowledge_base():
    if KB_PATH.exists():
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                st.session_state.knowledge_base = json.load(f)

            st.session_state.valid_message_numbers = {
                str(p.get('message_number', '')).strip()
                for p in st.session_state.knowledge_base
                if str(p.get('message_number', '')).strip()
            }
            return True
        except:
            return False
    return False

# >>> NEW: helper to build CSV snapshot of the case
def _build_case_csv_bytes(user_query: str) -> bytes:
    issue = st.session_state.get("current_issue") or {}
    meta = {
        "message_number": str(issue.get("message_number", "")),
        "message_text": (issue.get("message_text") or "").strip(),
        "reason": (issue.get("reason") or "").strip(),
        "solution": (issue.get("solution") or "").strip(),
        "note": (issue.get("note") or "").strip(),
        "escalated_at_utc": datetime.now(timezone.utc).isoformat(),
        "follow_up_attempts": str(st.session_state.get("follow_up_attempts", 0)),
    }

    buf = StringIO()
    w = csv.writer(buf)
    # Meta header section
    w.writerow(["field", "value"])
    for k, v in meta.items():
        w.writerow([k, v])
    # User last turn (trigger)
    w.writerow([])
    w.writerow(["trigger_user_query", user_query])

    # Conversation log
    w.writerow([])
    w.writerow(["role", "content"])
    for m in st.session_state.get("chat_history", []):
        role = m.get("role", "")
        content = (m.get("content", "") or "").replace("\n", " ").strip()
        # Avoid dumping raw image bytes; mark presence instead
        if m.get("images"):
            content = f"{content} [attached_image]"
        w.writerow([role, content])

    # UTF-8 with BOM so Arabic is Excel-friendly
    return ("\ufeff" + buf.getvalue()).encode("utf-8-sig")

# >>> NEW: upload to GCS and return gs:// URI (or None)
def _upload_case_to_gcs(user_query: str):
    if not GCS_BUCKET_NAME:
        return None  # not configured; skip quietly

    try:
        # Lazy import so your app runs even if google libs aren't installed
        from google.cloud import storage
        from google.oauth2 import service_account

        # Build client
        if GCS_CREDENTIALS_JSON and os.path.exists(GCS_CREDENTIALS_JSON):
            creds = service_account.Credentials.from_service_account_file(GCS_CREDENTIALS_JSON)
            client = storage.Client(credentials=creds, project=creds.project_id)
        else:
            client = storage.Client()  # ADC

        bucket = client.bucket(GCS_BUCKET_NAME)

        issue = st.session_state.get("current_issue") or {}
        msg_no = str(issue.get("message_number", "") or "unknown")
        uid = uuid4().hex[:8]
        day = datetime.utcnow().strftime("%Y/%m/%d")
        object_name = f"escalations/{day}/case_{msg_no}_{uid}.csv"

        data = _build_case_csv_bytes(user_query)
        blob = bucket.blob(object_name)
        blob.upload_from_string(data, content_type="text/csv")

        return f"gs://{GCS_BUCKET_NAME}/{object_name}"
    except Exception as e:
        # Don’t break the UX; just log a warning for admins/devs
        st.warning(f"تعذر رفع تقرير التصعيد إلى السحابة: {e}")
        return None

def _escalate_and_reset(user_query: str, preface: str = None):
    # Upload snapshot before resetting state
    gcs_uri = _upload_case_to_gcs(user_query)
    if gcs_uri:
        st.session_state.last_escalation_uri = gcs_uri

    parts = []
    if preface:
        parts.append(preface.strip())

    parts.append(
        "يبدو أنك محجوب عن إكمال الخطوات من هذه الرسالة. سأربطك الآن بالمستشار:\n"
        "- **البريد الإلكتروني:** support@example.com\n"
        "- **الهاتف:** 123-456-7890\n"
        "تم تسجيل الحالة لدينا ليطّلع عليها المستشار."
    )
    reply = "\n\n".join(parts)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.session_state.memory.save_context({"question": user_query}, {"output": reply})

    # reset state
    st.session_state.solution_provided = False
    st.session_state.active_troubleshooting = False
    st.session_state.follow_up_attempts = 0
    st.session_state.awaiting_anything_else = False
    st.session_state.current_issue = None
    st.session_state.current_issue_embedding = None
    st.rerun()

def to_english_digits(s: str) -> str:
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return s.translate(trans)

def get_text_embedding(text: str):
    model = get_embedding_model()
    return model.encode(text, convert_to_tensor=True)




@st.cache_resource
def create_anything_else_guard_chain():
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""
أنت مصنّف نية بعد إغلاق مشكلة سابقة. قرّر هل يريد المستخدم:
- فتح مشكلة/تذكرة جديدة الآن  => "open_new"
- إنهاء المحادثة (لا، شكراً، خلاص…) => "done"
- الرسالة غير واضحة             => "unclear"

أخرج JSON فقط:
{{
  "decision": "open_new" | "done" | "unclear",
  "new_issue_number": <string or null>,   // إن وُجد رقم رسالة في النص (أرقام عربية أو إنجليزية)
  "confidence": <float 0..1>
}}

إرشادات:
- إن ذكر المستخدم رقم رسالة (مثل 939 أو ١٤٩٥) فاعتبر القرار "open_new" واستخرج الرقم.
- صيغ مقبولة لـفتح جديد: "افتح تذكرة/مشكلة", "أبدأ مشكلة جديدة", "قضية جديدة", "start a new ticket/issue", "open new", "another error", إلخ.
- صيغ الإنهاء: "لا", "لا شكراً", "خلاص", "تم", "تمام", "thanks", "no thanks", إلخ.
- أمثلة سريعة:
  - "لا شكراً" -> done
  - "افتح تذكرة جديدة" -> open_new
  - "939" -> open_new + new_issue_number="939"
  - "نعم" بعد سؤال "هل تحتاج شيئاً آخر؟" -> open_new (إن لم تُذكر معلومة أخرى)
  - "شكراً" -> done

نص المستخدم:
\"\"\"{user_text}\"\"\"

أجب JSON فقط.
""",
        input_variables=["user_text"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

def get_issue_reference_embedding(issue: dict):
    if st.session_state.current_issue_embedding is not None:
        return st.session_state.current_issue_embedding
    ref_text = " ".join([
        issue.get("message_text", "") or "",
        issue.get("reason", "") or "",
        issue.get("solution", "") or ""
    ]).strip()
    emb = get_text_embedding(ref_text)
    st.session_state.current_issue_embedding = emb
    return emb

@st.cache_resource
def create_llm_passage_extractor_chain():
    """
    Takes a user query and a large document text and extracts the single
    most relevant passage that answers the query.
    """
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""You are an expert data extractor. You are given a user's question and the full text of a relevant document section. Your task is to find and extract the specific paragraph, sentence, or definition that directly answers the user's question from within the document text.

- Be precise. Only extract the text that is directly relevant.
- If the document contains a list of definitions (e.g., lines starting with a bullet point '•'), extract only the definition that matches the user's question.
- If no specific passage is a good match, return the first paragraph of the document as a general fallback.

USER'S QUESTION: "{query}"

DOCUMENT TEXT:
---
{document_body}
---

EXTRACTED PASSAGE:""",
        input_variables=["query", "document_body"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)



@st.cache_resource
def create_relevance_gate_chain():
    """
    Acts as a final judge. Determines if the top search result is truly
    relevant to the user's query.
    """
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""You are a relevance judge. Your task is to determine if the retrieved document is a good and direct answer to the user's question.

USER'S QUESTION: "{query}"

RETRIEVED DOCUMENT TITLE: "{title}"
RETRIEVED DOCUMENT SNIPPET: "{snippet}"

Based on the title and snippet, is this document a direct and relevant answer to the user's specific question?
Answer with a valid JSON object only, with the keys "is_relevant" (true or false) and "confidence" (a float from 0.0 to 1.0).
{{"is_relevant": true, "confidence": 0.9}}
""",
        input_variables=["query", "title", "snippet"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

# --- NEW: render answer from guide section (+ Solution Image attachments) ---
def render_guide_answer(section: dict, user_query: str = ""):
    """
    Upgraded version: First, uses an LLM to extract the most relevant passage
    from the selected section, then generates a formatted answer from that passage.
    """
    original_body = section.get("body", "")

    try:
        # Create a copy of the section to avoid modifying the original in session state
        modified_section = section.copy()

        # Call the new extractor chain
        extractor_chain = create_llm_passage_extractor_chain()
        extracted_passage = extractor_chain.invoke({
            "query": user_query,
            "document_body": original_body
        })["text"].strip()

        print(f"  -> Extracted Passage: {extracted_passage[:100]}...")

        # Replace the body in our temporary copy with the new, shorter passage
        modified_section["body"] = extracted_passage

        # Generate the final answer using the MODIFIED section
        data = generate_guide_instructions(modified_section, user_query=user_query)

    except Exception as e:
        print(f"  -> Passage extraction failed: {e}. Falling back to full text.")
        # If extraction fails for any reason, fall back to the original behavior
        data = generate_guide_instructions(section, user_query=user_query)

    # --- The rest of the function remains the same ---
    msg = {"role": "assistant", "content": data["answer"], "images": []}
    for p in (data.get("images") or []):
        msg["images"].append(p)
    if msg["content"].strip():
        msg["content"] += f"\n\n> المصدر: {section.get('title', 'دليل الأستاذ العام')}"

    st.session_state.chat_history.append(msg)

    # Reset the error context state
    st.session_state.solution_provided = False
    st.session_state.active_troubleshooting = False
    st.session_state.current_issue = None
    st.session_state.follow_up_attempts = 0
def classify_active_message(user_text: str, kb: list) -> str:
    text = to_english_digits((user_text or "").strip())
    if not (st.session_state.get("solution_provided") and st.session_state.get("current_issue")):
        return "other"
    resolved_cues = ["تم حلها","انحلت","اشتغلت","works now","solved","تمام","شكرا","ثانكس"]
    if any(c in text for c in resolved_cues):
        return "resolved"
    issue = st.session_state.current_issue
    curr_no = str(issue.get("message_number", "")).strip()
    nums = re.findall(r'\d+', text)
    if nums:
        if curr_no and all(n == curr_no for n in nums):
            return "follow_up"
        if curr_no and any(n != curr_no for n in nums):
            return "new_issue"
    followup_cues = ["لم أستطع","لم اقدر","مش واضح","غير واضح","كيف","لم يعمل","ما المقصود",
                     "ما زال","لا يزال","الخطوة","التالي","أعد","أعد الخطوة","فسّر","شرح"]
    if len(text) <= 60 or any(c in text for c in followup_cues):
        return "follow_up"
    try:
        query_emb = get_text_embedding(text)
        curr_emb = get_issue_reference_embedding(issue)
        s_current = util.cos_sim(query_emb, curr_emb).item()
        others = [p for p in kb if p is not issue and 'embedding' in p]
        if others:
            corpus = torch.tensor([p['embedding'] for p in others])
            s_vec = util.cos_sim(query_emb, corpus)[0]
            best_other_idx = int(s_vec.argmax().item())
            s_other = float(s_vec[best_other_idx].item())
        else:
            s_other = 0.0
        if (s_current >= ACTIVE_INTENT_THRESHOLDS["close_to_current"]) and \
           ((s_current - s_other) >= ACTIVE_INTENT_THRESHOLDS["margin"]):
            return "follow_up"
        if s_other >= ACTIVE_INTENT_THRESHOLDS["new_issue_candidate"] and s_other > s_current:
            return "new_issue"
        return "follow_up"
    except Exception:
        return "follow_up"

def semantic_search(query: str):
    kb = st.session_state.get('knowledge_base', [])
    if not kb:
        return {"answer": "قاعدة المعرفة فارغة."}

    # 1) number-first routing
    query = to_english_digits(query or "")
    query_numbers = re.findall(r'\d+', query)
    if query_numbers:
        # exact number in KB?
        for problem in kb:
            kb_number = str(problem.get('message_number', '')).strip()
            if kb_number and kb_number in query_numbers:
                return problem
        # number mentioned but not found → ask a human
        return {"not_found": True, "reason": "number_not_in_kb", "number": query_numbers[0]}

    # 2) semantic + lexical gating
    embedding_model = get_embedding_model()
    corpus_embeddings = torch.tensor([p['embedding'] for p in kb])
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    best_idx = int(torch.argmax(cos_scores).item())
    best_score = float(cos_scores[best_idx].item())
    best_match = kb[best_idx]

    # lexical signals against the best candidate
    title = (best_match.get("message_text") or "")[:300]
    lex = _lexical_ratio(query, title)
    cov = _token_coverage(query, title)

    # thresholds (tuned conservatively)
    SEM_ACCEPT = 0.66     # accept as direct answer above this
    SEM_SUGGEST = 0.58    # consider "did you mean?" above this
    LEX_ACCEPT = 0.80     # strong lexical
    LEX_SUGGEST = 0.70    # weaker but still plausible
    COV_ACCEPT = 0.80     # token overlap coverage
    COV_SUGGEST = 0.60

    # Accept only if clearly close on either semantic or lexical
    if best_score >= SEM_ACCEPT or lex >= LEX_ACCEPT or cov >= COV_ACCEPT:
        return best_match

    # Offer "did you mean?" only when there’s some convincing overlap
    if (best_score >= SEM_SUGGEST) and (lex >= LEX_SUGGEST or cov >= COV_SUGGEST):
        return {"suggestion": best_match}

    # Otherwise, don't risk a wrong answer → handoff
    return {"not_found": True, "reason": "low_similarity"}


def log_user_feedback(query: str, chosen_title: str, feedback_file: str = "feedback_log.csv"):
    """Appends a query and the user's chosen correct answer to a CSV log file."""
    try:
        # Create file and header if it doesn't exist
        if not os.path.exists(feedback_file):
            with open(feedback_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_utc", "query", "chosen_title"])

        # Append the new feedback
        with open(feedback_file, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(timezone.utc).isoformat(), query, chosen_title])
        print(f"Feedback logged: '{query}' -> '{chosen_title}'")
    except Exception as e:
        print(f"Error logging feedback: {e}")


def _ensure_kb_matrices():
    # Build tensors once (for speed) after kb loads
    if st.session_state.get("gl_errors_kb") and "gl_errors_matrix" not in st.session_state:
        st.session_state.gl_errors_matrix = torch.tensor([p["embedding"] for p in st.session_state.gl_errors_kb])
    if st.session_state.get("gl_guide_kb") and "gl_guide_matrix" not in st.session_state:
        st.session_state.gl_guide_matrix = torch.tensor([s["embedding"] for s in st.session_state.gl_guide_kb])


def _lexical_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    def normalize(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


from langchain_google_genai import \
    GoogleGenerativeAIEmbeddings  # <-- Make sure to add this import at the top of your file





# @st.cache_resource
# def create_keyword_extraction_chain():
#     """
#     Creates an LLM chain that extracts key technical nouns from a user's query.
#     """
#     llm = get_llm_model()
#     prompt = PromptTemplate(
#         template="""You are an expert search pre-processor. Your task is to extract the key nouns and technical terms from the user's query. Ignore conversational words like 'how to', 'what is', 'explain', etc. Respond with only the extracted terms.
#
# User Query: "{query}"
#
# Example 1:
# User Query: "كيف اضيف سند قبض لعميل"
# Extracted Terms: "سند قبض عميل"
#
# Example 2:
# User Query: "ما هو تقرير ميزان المراجعة"
# Extracted Terms: "تقرير ميزان المراجعة"
#
# Now, extract the key terms from the user's query above.
# Extracted Terms:""",
#         input_variables=["query"],
#     )
#     return LLMChain(llm=llm, prompt=prompt, verbose=False)




def llm_search_gl_guide(query: str, k_titles: int = 12, top_m: int = 3):
    """
    Pure LLM retrieval:
    - Round 1: feed ALL titles to LLM → pick up to k_titles ids.
    - Round 2: feed snippets of those to LLM → get top_m ranked ids.
    - Return same shape your UI expects: {"candidates": [sections...]}
    """
    kb = st.session_state.get("gl_guide_kb", []) or []
    if not kb:
        return {"answer": "دليل المستخدم (GL) غير محمّل بعد."}

    # ---- Round 1: titles only ----
    lines = []
    for i, sec in enumerate(kb):
        title = (sec.get("title") or "").strip().replace("\n", " ")
        cat   = (sec.get("category") or "").strip()
        lines.append(f"{i} | {title} | {cat}")
    items_blob = "\n".join(lines)

    sel_chain = create_llm_title_selector_chain()
    try:
        r1_raw = sel_chain.invoke({"query": query, "items": items_blob, "k": k_titles})["text"]
        r1 = _json_only(r1_raw)
        ids1 = [int(x) for x in (r1.get("ids") or []) if 0 <= int(x) < len(kb)]
    except Exception:
        # fallback: naive lexical pick of first 12
        ids1 = list(range(min(k_titles, len(kb))))

    if not ids1:
        return {"answer": "عذراً، لم أتمكن من العثور على قسم مطابق."}

    # ---- Round 2: snippets for the selected ----
    cand_lines = []
    for i in ids1:
        sec = kb[i]
        title = (sec.get("title") or "").strip().replace("\n", " ")
        cat   = (sec.get("category") or "").strip()
        body  = (sec.get("body") or "").strip().replace("\r", " ")
        snippet = body[:480].replace("\n", " ")
        cand_lines.append(f"{i} | {title} | {cat} | {snippet}")
    cands_blob = "\n".join(cand_lines)

    rank_chain = create_llm_candidate_ranker_chain()
    try:
        r2_raw = rank_chain.invoke({"query": query, "cands": cands_blob})["text"]
        r2 = _json_only(r2_raw)
        ranked = [int(x) for x in (r2.get("ranked_ids") or []) if x in ids1]
    except Exception:
        ranked = ids1[:top_m]

    if not ranked:
        ranked = ids1[:top_m]

    # keep top_m sections
    out = [kb[i] for i in ranked[:top_m]]
    return {"candidates": out}

@st.cache_resource
def create_llm_errors_selector_chain():
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""
اختر أفضل عنصر رسالة خطأ يطابق سؤال المستخدم.
أولوية مطابقة الرقم إن وُجد صراحةً.

سؤال المستخدم:
{query}

العناصر (id | message_number | message_text | reason):
{items}

JSON فقط:
{{"best_id": <int or -1>, "confidence": 0.0_to_1.0}}
""",
        input_variables=["query","items"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

def search_gl_errors_llm(query: str):

    kb = st.session_state.get("gl_errors_kb", []) or []
    if not kb:
        return {"answer":"ملف رسائل الأخطاء غير محمّل بعد."}

    # prefer IDs detected by the smart extractor
    detected = st.session_state.pop("__detected_error_ids__", [])
    if detected:
        for n in detected:
            for p in kb:
                if str(p.get("message_number", "")).strip() == n:
                    return p

    q = to_english_digits(query or "")
    # 1) exact number short-circuit
    nums = re.findall(r"\d{3,}", q)
    if nums:
        nset = set(nums)
        for i,p in enumerate(kb):
            if str(p.get("message_number","")).strip() in nset:
                return p

    # 2) LLM shortlist
    items = "\n".join(
        f"{i} | {p.get('message_number','')} | {(p.get('message_text') or '').strip()} | {(p.get('reason') or '').strip()}"
        for i,p in enumerate(kb)
    )
    raw = create_llm_errors_selector_chain().invoke({"query": query, "items": items})["text"]
    data = _json_only(raw) or {}
    i = int(data.get("best_id", -1))
    return kb[i] if 0 <= i < len(kb) else {"not_found": True, "reason": "no_match"}




@st.cache_resource
def create_intent_category_chain():
    """
    Creates an LLM chain that classifies the user's query into one of the four
    business categories.
    """
    llm = get_llm_model()
    prompt = PromptTemplate(
        template="""You are an expert ERP system router. Your job is to classify the user's query into one of four categories based on their likely intent. The categories are:
- Configuration: For system administrators setting up core system variables. High-stakes, foundational questions.
- Inputs: For users creating or defining master data (e.g., "how to add a bank", "create a cash fund").
- Operations: For users performing a financial transaction (e.g., "issue a receipt voucher", "reconcile a bank account").
- Reports: For managers or users analyzing data and outcomes (e.g., "show me the trial balance", "what is the profit and loss report?").

User Query: "{query}"

Based on the user's query, which of the four categories is the most likely target?
Respond with JSON only, like this: {{"category": "<Configuration|Inputs|Operations|Reports>"}}""",
        input_variables=["query"],
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "kb_loaded" not in st.session_state:
        st.session_state.kb_loaded = False
    if "rejection_count" not in st.session_state:
        st.session_state.rejection_count = 0

    if "is_in_rephrase_loop" not in st.session_state:
        st.session_state.is_in_rephrase_loop = False
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=False
        )

    # Placeholders
    if "gl_guide_kb" not in st.session_state:
        st.session_state.gl_guide_kb = []
    if "gl_errors_kb" not in st.session_state:
        st.session_state.gl_errors_kb = []



    # ✅ Load all KBs (legacy → guide → errors). Errors last to set valid_message_numbers.
    legacy_ok = load_knowledge_base()
    guide_ok  = load_gl_guide_kb()

    _ensure_kb_matrices()
    build_gl_dyk_index()
    errors_ok = load_errors_kb()

    st.session_state.kb_loaded = legacy_ok or guide_ok or errors_ok or st.session_state.kb_loaded

    if "processed_image_id" not in st.session_state:
        st.session_state.processed_image_id = None
    if "awaiting_more" not in st.session_state:
        st.session_state.awaiting_more = False
    if "current_issue" not in st.session_state:
        st.session_state.current_issue = None
    if "current_issue_embedding" not in st.session_state:
        st.session_state.current_issue_embedding = None
    if "active_troubleshooting" not in st.session_state:
        st.session_state.active_troubleshooting = False
    if "awaiting_anything_else" not in st.session_state:
        st.session_state.awaiting_anything_else = False
    if "solution_provided" not in st.session_state:
        st.session_state.solution_provided = False
    if "follow_up_attempts" not in st.session_state:
        st.session_state.follow_up_attempts = 0
    if "pending_suggestion" not in st.session_state:
        st.session_state.pending_suggestion = None
    if "pending_choices" not in st.session_state:
        st.session_state.pending_choices = None

@st.cache_resource
def create_followup_policy_chain():
    llm = get_llm_model()
    policy_prompt = """
أنت "مشرف متابعة" ذكي لمساعد Onyx ERP.
لديك:
- المشكلة الحالية: رقم {message_number} / نص: {message_text}
- السبب: {reason}
- الحلّ الكامل (مصدر الحقيقة لخطوات الإصلاح): {solution}
- آخر ردَّين للمساعد: [1] {last_assistant_1}  |  [2] {last_assistant_2}
- آخر ردَّين للمستخدم: [1] {last_user_1}  |  [2] {last_user_2}
- رسالة المستخدم الحالية: {user_text}

مهمتك: قرِّر الإجراء التالي بشكل عام (دون الاعتماد على قواعد خاصة بكل خطأ):
- "step_help": يمكنك تقديم **خطوة جديدة ومميّزة وقابلة للتنفيذ** مستخلصة من الحل أعلاه، ولم تُذكر في ردود المساعد الأخيرة.
- "ask_clarify": تحتاج معلومة محددة مفقودة لتحديد الخطوة التالية (اكتب سؤالًا واحدًا دقيقًا فقط).
- "escalate": المستخدم يبدو محجوبًا/مقيّد صلاحيات/لا يصل للشاشة اللازمة/جرَّب ولم ينجح/أو لا توجد خطوة جديدة واضحة من الحل يمكنك تقديمها بدون تكرار. في هذه الحالة يجب التصعيد فورًا لمستشار.

أجب بصيغة JSON فقط:
{{
  "action": "step_help" | "ask_clarify" | "escalate",
  "assistant_reply": "<النص الذي سأقوله للمستخدم إذا كانت action لا تساوي escalate>",
  "confidence": 0.0
}}
"""
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=policy_prompt,
            input_variables=[
                "message_number","message_text","reason","solution",
                "last_assistant_1","last_assistant_2","last_user_1","last_user_2",
                "user_text"
            ]
        ),
        verbose=False
    )

@st.cache_resource
def create_conversational_chain():
    llm = get_llm_model()
    prompt_template = """
    أنت مساعد دعم فني متخصص فقط في نظام "Onyx ERP". مهمتك هي مساعدة المستخدمين الذين فشل بحثهم الأولي.
    اتبع هذه القواعد الصارمة:
    1.  **لا تكن عامًا أبدًا**: لا تسأل عن "نوع الجهاز" أو "نظام التشغيل".
    2.  **إذا كان سؤال المستخدم غامضًا**: اطلب منه بأدب تقديم وصف أكثر تحديدًا للمشكلة أو رقم رسالة الخطأ.
    3.  **إذا قال المستخدم أن الحل لم ينجح**: اطرح سؤالاً توضيحيًا حول الحل السابق.
    4.  **كن موجزًا ومباشرًا**.
    ---
    سجل المحادثة: {chat_history}
    سؤال المستخدم الحالي: {question}
    إجابتك المركزة والمختصرة:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "question"])
    return LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory, verbose=True)

def render_sidebar():
    with st.sidebar:
        st.file_uploader("أو قم بتحميل صورة لرسالة الخطأ", type=["png", "jpg", "jpeg"], key="image_uploader")
        st.markdown("---")
        with st.expander("🧠 إعداد الأدلة (GL فقط الآن)"):
            st.caption("سيتم استخدام الملفات الافتراضية التالية ما لم تغيِّر المسارات:")
            guide_pdf = st.text_input("مسار دليل المستخدم (GL)", GUIDE_PDF_DEFAULT)
            errors_pdf = st.text_input("مسار ملف الأخطاء والحلول", ERRORS_PDF_DEFAULT)

            colA, colB = st.columns(2)
            if colA.button("بناء دليل المستخدم (GL)", type="primary"):
                if not os.path.exists(guide_pdf):
                    st.error(f"الملف غير موجود: {guide_pdf}")
                elif build_gl_guide_kb(guide_pdf):
                    load_gl_guide_kb()
                    st.success("تم بناء دليل المستخدم (GL).")

            if colB.button("بناء ملف الأخطاء (GL)"):
                if not os.path.exists(errors_pdf):
                    st.error(f"الملف غير موجود: {errors_pdf}")
                elif build_errors_kb(errors_pdf):
                    load_errors_kb()
                    st.success("تم بناء ملف الأخطاء (GL).")

def render_welcome_screen():
    st.title("🤖 أهلاً بك في المساعد الذكي")
    st.markdown("### للبدء، يرجى بناء قاعدة المعرفة أولاً من الشريط الجانبي.")


def render_suggestion_buttons():
    suggestion_data = st.session_state.pending_suggestion

    if "message_text" in suggestion_data:
        suggestion_text = suggestion_data['message_text']
    elif "title" in suggestion_data:
        suggestion_text = suggestion_data['title']
    else:
        suggestion_text = "اقتراح غير معروف"

    st.info(f'هل تقصد هذا الموضوع؟ "{suggestion_text}"')

    col1, col2 = st.columns(2)
    if col1.button("نعم", key="confirm_suggestion", use_container_width=True):
        confirmed_item = st.session_state.pending_suggestion
        st.session_state.pending_suggestion = None

        if "message_number" in confirmed_item:
            handle_confirmed_solution(confirmed_item)
        elif "body" in confirmed_item:
            last_user_query = ""
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "user":
                    last_user_query = msg["content"]
                    break
            render_guide_answer(confirmed_item, user_query=last_user_query)
        st.rerun()

    if col2.button("لا", key="reject_suggestion", use_container_width=True):
        st.session_state.pending_suggestion = None
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "حسنًا، هل يمكنك وصف ما تبحث عنه بمزيد من التفصيل؟"})
        st.rerun()


def render_choice_buttons():
    """
    Renders choices and logs the user's selection for future training.
    """
    st.info("لقد وجدت بعض المواضيع التي قد تكون ذات صلة. يرجى اختيار الأنسب:")

    candidate_sections = st.session_state.pending_choices

    for i, section in enumerate(candidate_sections):
        if st.button(section['title'], key=f"choice_{i}", use_container_width=True):
            last_user_query = ""
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "user":
                    last_user_query = msg["content"]
                    break


            st.session_state.pending_choices = None
            render_guide_answer(section, user_query=last_user_query)
            st.rerun()

    if st.button("لا شيء من هذه الخيارات", key="reject_choices", use_container_width=True):
        st.session_state.pending_choices = None
        st.session_state.rejection_count += 1  # Increment the counter

        if st.session_state.rejection_count >= 2:
            print("Search failed twice. Escalating to advisor.")

            # Find the last user query that caused the failure
            last_user_query = ""
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "user":
                    last_user_query = msg["content"]
                    break

            _escalate_and_reset(last_user_query, preface="لم أتمكن من العثور على إجابة مناسبة في دليل المستخدم.")
            return

        else:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "حسنًا، هل يمكنك وصف ما تبحث عنه بكلمات مختلفة أو بمزيد من التفصيل؟"})

        st.rerun()

def render_chat_interface():
    """Renders the main chat UI and handles the interaction logic."""
    st.title("الوكيل الذكي لنظام Onyx ERP")

    if not st.session_state.chat_history:
        st.markdown("<div style='text-align: center;'>أهلاً بك! أنا مساعدك الذكي لحل مشاكل نظام Onyx ERP.</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: grey;'>يمكنك البدء باختيار أحد الأسئلة الشائعة أدناه.</div><br>", unsafe_allow_html=True)

        cols = st.columns([1, 1, 1])
        if cols[0].button("حل مشكلة 'صنف منتهي' (1495)", use_container_width=True):
            handle_text_query("1495"); st.rerun()
        if cols[1].button("حل مشكلة 'تهيئة العملات' (669)", use_container_width=True):
            handle_text_query("669"); st.rerun()
        if cols[2].button("حل مشكلة 'كميات مجانية' (939)", use_container_width=True):
            handle_text_query("939"); st.rerun()

    for msg in st.session_state.chat_history:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if "images" in msg and msg.get("images"):
                for img_data in msg["images"]:
                    st.image(img_data, use_container_width=True)

    chat_input_disabled = bool(st.session_state.pending_suggestion)

    if st.session_state.pending_suggestion:
        render_suggestion_buttons()
    elif st.session_state.pending_choices:
        render_choice_buttons()

    chat_input_disabled = bool(st.session_state.pending_suggestion or st.session_state.pending_choices)

    if st.session_state.solution_provided:
        if st.button("✅ تم حلها", use_container_width=True):
            st.session_state.solution_provided = False
            st.session_state.follow_up_attempts = 0
            st.session_state.active_troubleshooting = False
            st.session_state.current_issue = None
            st.session_state.current_issue_embedding = None
            st.session_state.chat_history.append({"role": "assistant", "content": "رائع! يسعدني أنني تمكنت من المساعدة."})
            st.rerun()

    if st.session_state.get("image_uploader") and st.session_state.image_uploader.file_id != st.session_state.processed_image_id:
        handle_image_upload()

    if user_query := st.chat_input("ما هو الخطأ الذي تواجهه؟", disabled=chat_input_disabled):
        handle_text_query(user_query)

def handle_image_upload():
    uploaded_image = st.session_state.image_uploader
    st.session_state.processed_image_id = uploaded_image.file_id
    image_bytes = uploaded_image.getvalue()
    st.session_state.chat_history.append({"role": "user", "content": f"تم تحميل صورة: `{uploaded_image.name}`", "images": [image_bytes]})

    if st.session_state.solution_provided:
        interception_message = """
        لحظة من فضلك، يبدو أننا ما زلنا نعمل على حل المشكلة السابقة.

        - إذا تم حل المشكلة، يرجى الضغط على زر **"✅ تم حلها"** الأخضر.
        - إذا كنت لا تزال تواجه صعوبة، من فضلك أخبرني حتى نتمكن من المتابعة أو توجيهك إلى مستشار.
        """
        st.session_state.chat_history.append({"role": "assistant", "content": interception_message})
        st.rerun()
        return

    with st.spinner("🔍 تحليل الصورة..."):
        extracted_number = llm_extract_message_number_from_image(image_bytes)

    if extracted_number:
        handle_text_query(extracted_number)
    else:
        st.session_state.chat_history.append({"role": "assistant",
                                              "content": "عذراً، لم أتمكن من استخراج رقم الرسالة من الصورة. حاول وصف المشكلة كتابياً."})
        st.rerun()

def handle_confirmed_solution(problem_data):
    st.session_state.solution_provided = True
    st.session_state.follow_up_attempts = 0
    response_data = generate_instructional_response(problem_data)

    st.session_state.current_issue = problem_data
    st.session_state.current_issue_embedding = None
    st.session_state.active_troubleshooting = True

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_data.get("answer"),
        "images": response_data.get("images", [])
    })
    st.session_state.memory.save_context(
        {"question": problem_data['message_text']},
        {"output": response_data.get("answer", "")}
    )

def handle_text_query(user_query):
    """
    Search -> Format -> Converse flow with:
    - post-close guard ("anything else?" flow) + open_new intent
    - hard handoff if the user explicitly asks for a human
    - PERMISSION hotword override -> escalate with preface
    - active-issue router (follow_up/new_issue/open_new/resolved/other)
    - "did you mean?" confirmation for partial titles (<95% lexical match)
    - immediate escalation if user typed a non-existent error number (errors-only)
    - NEW: guide/errors source routing + attach GL guide 'Solution Image' pictures
    """
    import difflib
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # ---------------- helpers ----------------
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _normalize_for_title(s: str) -> str:
        s = _norm(s)
        s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _lexical_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, _normalize_for_title(a), _normalize_for_title(b)).ratio()

    def _looks_like_error_intent(s: str) -> bool:
        t = (s or "").strip().lower()
        if re.fullmatch(r"\d{3,}", re.sub(r"\s+", "", to_english_digits(t))):
            return True
        if re.search(r"\b(error|code|message|msg|رسالة|خطأ|مشكلة)\b", t):
            return True
        return bool(re.search(r"\d{3,}", t))

    def _simple_open_new(s: str) -> bool:
        t = _norm(s)
        return bool(re.search(r"(افتح|فتح|ابدأ|بدء)\s+(مشكلة|تذكرة)|open\s+new\s+(issue|ticket)", t))

    # ===== 0) GLOBAL PERMISSION OVERRIDE =====
    if _is_permission_related({}, user_query):
        preface = (
            "🔒 **المشكلة مرتبطة بالصلاحيات**.\n"
            "إذا لم تكن تمتلك الصلاحية المطلوبة، رجاءً تواصل مع **مسؤول النظام** أو **مديرك** "
            "لمنحك الإذن المناسب على الشاشة/الإجراء المطلوب.\n"
            "سأقوم الآن بإغلاق هذه المشكلة (إن وُجدت) وتحويلك إلى مستشار لمتابعة منح الصلاحية."
        )
        _escalate_and_reset(user_query, preface=preface)
        return

    # ===== 1) HARD HANDOFF: explicit ask for human =====
    if _wants_handoff(user_query):
        _escalate_and_reset(user_query)
        return

    # ===== 2) POST-CLOSE GUARD ("anything else?" mode) =====
    if st.session_state.get("awaiting_anything_else"):
        router = create_intent_router_chain()
        ctx = {
            "current_number": "",
            "current_text": "",
            "current_reason": "",
            "current_solution": "",
            "last_assistant_msg": _last_assistant_message_text(),
            "valid_numbers": _numbers_catalog(),
            "user_text": user_query,
        }
        try:
            decision_raw = router.invoke(ctx)["text"]
            decision = _json_from_model_text(decision_raw)
        except Exception:
            decision = {"intent": "other", "new_issue_number": None, "confidence": 0.4}

        intent = decision.get("intent", "other")
        new_no = decision.get("new_issue_number")

        if intent == "other" and _simple_open_new(user_query):
            intent = "open_new"

        if intent == "open_new":
            st.session_state.awaiting_anything_else = False
            reply = "تم 👍 — أرسل رقم الرسالة أو وصفًا قصيرًا للمشكلة الجديدة."
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.session_state.memory.save_context({"question": user_query}, {"output": reply})
            return st.rerun()

        if intent == "new_issue" and new_no:
            st.session_state.awaiting_anything_else = False
            with st.spinner("🔎 جاري فتح المشكلة الجديدة..."):
                sr = search_gl_errors_llm(str(new_no))
                if "message_number" in sr:
                    st.session_state.solution_provided = True
                    st.session_state.follow_up_attempts = 0
                    st.session_state.current_issue = sr

                    st.session_state.active_troubleshooting = True
                    resp = generate_instructional_response(sr)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": resp.get("answer"),
                        "images": resp.get("images", [])
                    })
                    st.session_state.memory.save_context({"question": user_query}, {"output": resp.get("answer", "")})
                    return st.rerun()
                else:
                    st.session_state.chat_history.append({"role": "assistant",
                        "content": "لم أجد رقم الرسالة المذكور في الدليل. سأوصلك بمستشار الآن."})
                    _escalate_and_reset(user_query)
                    return

        reply = "هل تريد فتح مشكلة جديدة أم كل شيء تمام الآن؟"
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state.memory.save_context({"question": user_query}, {"output": reply})
        return st.rerun()

    # ===== 3) ACTIVE-ISSUE BRANCH =====
    if st.session_state.get("solution_provided"):
        kb = st.session_state.get("knowledge_base", []) or []
        issue = st.session_state.get("current_issue") or {}
        router = create_intent_router_chain()
        ctx = {
            "current_number": str(issue.get("message_number", "")),
            "current_text": (issue.get("message_text", "") or "")[:800],
            "current_reason": (issue.get("reason", "") or "")[:600],
            "current_solution": (issue.get("solution", "") or "")[:1200],
            "last_assistant_msg": _last_assistant_message_text(),
            "valid_numbers": _numbers_catalog(),
            "user_text": user_query
        }
        try:
            decision_raw = router.invoke(ctx)["text"]
            decision = _json_from_model_text(decision_raw)
        except Exception:
            decision = {"intent": "follow_up", "step_number": None, "new_issue_number": None, "confidence": 0.5}

        intent = decision.get("intent", "other")
        new_no = decision.get("new_issue_number")
        step_no = decision.get("step_number")

        if intent == "other" and _simple_open_new(user_query):
            intent = "open_new"

        if intent == "resolved":
            st.session_state.solution_provided = False
            st.session_state.follow_up_attempts = 0
            st.session_state.active_troubleshooting = False
            st.session_state.current_issue = None
            st.session_state.current_issue_embedding = None
            st.session_state.awaiting_anything_else = True
            st.session_state.chat_history.append({"role": "assistant", "content": "رائع! تم إغلاق المشكلة. هل تحتاج شيئًا آخر؟"})
            return st.rerun()

        if intent == "new_issue" and new_no:
            st.session_state.chat_history.append({"role": "assistant", "content":
                f"ذكرت رقم رسالة مختلف **{new_no}**. إذا انتهت المشكلة الحالية اضغط **\"✅ تم حلها\"** ثم أرسل الرقم الجديد لنفتحها."})
            return st.rerun()

        if intent == "open_new":
            st.session_state.awaiting_anything_else = True
            reply = (
                "لحظة من فضلك — ما زلنا نعمل على المشكلة الحالية.\n"
                "• إذا تم حلها، اضغط زر **\"✅ تم حلها\"** الأخضر.\n"
                "• إذا أردت فتح مشكلة جديدة الآن، أخبرني بإغلاق الحالية أو أرسل **رقم الرسالة/وصف المشكلة الجديدة**."
            )
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.session_state.memory.save_context({"question": user_query}, {"output": reply})
            return st.rerun()

        if intent == "follow_up":

            st.session_state.active_troubleshooting = True
            st.session_state.follow_up_attempts += 1

            # --- NEW ROUTER LOGIC ---
            print("--- 🧠 Follow-up Router Initiated ---")
            followup_router = create_followup_router_chain()
            solution_context = (issue.get("solution") or "")[:1000]  # Provide context from the solution

            try:
                router_raw = followup_router.invoke({
                    "solution_context": solution_context,
                    "query": user_query
                })["text"]
                followup_intent = _json_only(router_raw).get("intent", "troubleshoot_error")
                print(f"  -> Follow-up intent detected: {followup_intent}")
            except Exception as e:
                print(f"  -> Follow-up router failed: {e}. Defaulting to troubleshoot_error.")
                followup_intent = "troubleshoot_error"

            if followup_intent == "search_guide":
                # Path B: The user asked a "how-to" question. Search the guide.
                with st.spinner("🔎 بالبحث في دليل المستخدم..."):
                    res = llm_search_gl_guide(user_query)
                    if "candidates" in res and res["candidates"]:
                        best_choice = res["candidates"][0]
                        render_guide_answer(best_choice, user_query)
                        st.session_state.follow_up_attempts -= 1  # This successful lookup doesn't count as a "failed" attempt
                    else:
                        st.session_state.chat_history.append({"role": "assistant",
                                                              "content": "بحثت في دليل المستخدم ولكني لم أجد إجابة دقيقة. هل يمكنك إعادة صياغة سؤالك؟"})
            else:  # Path A: Default to troubleshooting the error
                # This is your existing, powerful troubleshooting logic
                pchain = create_followup_policy_chain()
                msgs = _last_msgs(2)
                raw = pchain.invoke({
                    "message_number": str(issue.get("message_number", "")),
                    "message_text": issue.get("message_text", ""),
                    "reason": issue.get("reason", ""),
                    "solution": issue.get("solution", ""),
                    "last_assistant_1": msgs["a1"], "last_assistant_2": msgs["a2"],
                    "last_user_1": msgs["u1"], "last_user_2": msgs["u2"],
                    "user_text": user_query
                })["text"]
                # ... (the rest of your original follow-up logic remains the same)
                try:
                    policy = _json_from_model_text(raw)
                except Exception:
                    policy = {"action": "ask_clarify", "assistant_reply": "ما الخطوة التي فشلت لديك تحديدًا؟"}

                action = policy.get("action")
                reply = (policy.get("assistant_reply") or "").strip()

                if action == "escalate":
                    _escalate_and_reset(user_query)
                    return

                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.session_state.memory.save_context({"question": user_query}, {"output": reply})

            if st.session_state.follow_up_attempts >= MAX_FOLLOW_UP_ATTEMPTS:
                _escalate_and_reset(user_query)
                return

            else:
                chain = create_troubleshooting_chain()
                reply = chain.predict(
                    message_number=str(issue.get("message_number", "")),
                    message_text=issue.get("message_text", ""),
                    reason=issue.get("reason", ""),
                    solution=issue.get("solution", ""),
                    chat_history=(st.session_state.memory.buffer if hasattr(st.session_state.memory, "buffer") else ""),
                    question=(user_query if step_no is None else f"الخطوة المقصودة هي رقم {step_no}. ما التالي؟")
                ).strip()

            st.session_state.memory.save_context({"question": user_query}, {"output": reply})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

            if st.session_state.follow_up_attempts >= MAX_FOLLOW_UP_ATTEMPTS or "نحتاج تصعيد" in reply:
                _escalate_and_reset(user_query)
                return

            return st.rerun()

        st.session_state.chat_history.append({"role": "assistant", "content":
            "هل يمكنك توضيح ما الذي لم ينجح تحديدًا في المشكلة الحالية، أم تريد فتح مشكلة جديدة برقم رسالة مختلف؟"})
        return st.rerun()

    # ===== 4) NEW-CASE SEARCH (no active issue) =====
    st.session_state.memory.save_context({"question": user_query}, {"output": ""})

    # Decide source first (so numbers inside how-to queries don't trigger false escalations)
    source = decide_source_smart(user_query)

    # If user explicitly asks to open new (no active issue), prompt for number/description
    if _simple_open_new(user_query):
        reply = "تم 👍 — أرسل رقم الرسالة أو وصفًا قصيرًا للمشكلة الجديدة."
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state.memory.save_context({"question": user_query}, {"output": reply})
        return st.rerun()

    # Errors-only: escalate if user typed a number that isn't in KB
    if source == "errors":
        valid_nums = st.session_state.get("valid_message_numbers", set()) or set()
        mentioned_nums = re.findall(r"\d+", user_query)
        if mentioned_nums and not any(n in valid_nums for n in mentioned_nums):
            st.session_state.chat_history.append({"role": "assistant",
                "content": "لم أجد رقم الرسالة الذي ذكرتَه في الدليل. سأوصلك بمستشار الآن لمساعدتك بشكل أدق."})
            _escalate_and_reset(user_query)
            return

    with st.spinner("🤔 أفكر..."):
        # Inside handle_text_query, within the `with st.spinner(...)` block:
        # Inside handle_text_query, within the `with st.spinner(...)` block:

        if source == "guide":
            res = llm_search_gl_guide(user_query)

            if "candidates" in res and res["candidates"]:
                if "candidates" in res and res["candidates"]:

                    st.session_state.rejection_count = 0
                    top_candidate = res["candidates"][0]
                    relevance_gate = create_relevance_gate_chain()


                    try:
                        snippet = (top_candidate.get("body") or "")[:400]
                        gate_raw = relevance_gate.invoke({
                            "query": user_query,
                            "title": top_candidate.get("title", ""),
                            "snippet": snippet
                        })["text"]

                        gate_result = _json_only(gate_raw)
                        is_relevant = gate_result.get("is_relevant", False)
                        confidence = gate_result.get("confidence", 0.0)

                        print(f"  -> Relevance Gate Check: Relevant={is_relevant}, Confidence={confidence:.2f}")

                        # Only proceed if the top result is deemed relevant with high confidence
                        if is_relevant and confidence >= 0.7:
                            st.session_state.pending_choices = res["candidates"]
                            st.session_state.pending_suggestion = None
                        else:
                            # If not relevant, provide a polite "I don't know" message
                            st.session_state.chat_history.append({"role": "assistant",
                                                                  "content": "عذراً، لم أجد إجابة مباشرة لسؤالك في دليل المستخدم. هل يمكنك طرح السؤال بطريقة أخرى؟"})
                    except Exception as e:
                        print(f"Relevance Gate failed: {e}")
                        # Fallback to showing choices if the gate itself fails
                        st.session_state.pending_choices = res["candidates"]


                else:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": res.get("answer", "حدث خطأ ما.")})

                st.rerun()

                st.session_state.pending_choices = res["candidates"]
                st.session_state.pending_suggestion = None  # Ensure old state is clear
                st.rerun()
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": res.get("answer", "حدث خطأ ما.")})
                st.rerun()

        # The 'else:' block for errors remains untouched and will continue to correctly
        # set st.session_state.pending_suggestion when needed.
        else:

            # Errors KB path

            sr = search_gl_errors_llm(user_query)

            if isinstance(sr, dict) and "message_number" in sr:

                kb_title = sr.get("message_text") or ""

                mentioned_nums = re.findall(r"\d+", to_english_digits(user_query))

                title_like = (len(_normalize_for_title(user_query)) <= 64) or _looks_like_error_intent(user_query)

                ratio = _lexical_ratio(user_query, kb_title)

                # Keep your "did you mean?" rule: only if user didn't provide a number

                if title_like and ratio < 0.95 and not mentioned_nums:
                    st.session_state.pending_suggestion = sr

                    return st.rerun()

                # Accept and answer

                st.session_state.solution_provided = True

                st.session_state.follow_up_attempts = 0

                st.session_state.current_issue = sr

                st.session_state.current_issue_embedding = (

                    torch.tensor(sr.get("embedding", []), dtype=torch.float32)

                    if sr.get("embedding") else None

                )

                st.session_state.active_troubleshooting = True

                resp = generate_instructional_response(sr)

                st.session_state.chat_history.append({

                    "role": "assistant",

                    "content": resp.get("answer"),

                    "images": resp.get("images", [])

                })

                st.session_state.memory.save_context({"question": user_query}, {"output": resp.get("answer", "")})

                return st.rerun()

            # ❗ Nothing matched in errors KB

            if re.search(r"\d{3,}", to_english_digits(user_query)):
                # User explicitly gave a number → escalate now

                st.session_state.chat_history.append({"role": "assistant",

                                                      "content": "لم أجد رقم الرسالة الذي ذكرتَه في الدليل. سأوصلك بمستشار الآن لمساعدتك بشكل أدق."})

                _escalate_and_reset(user_query)

                return

            # Otherwise fallback to conversational

            chain = create_conversational_chain()

            reply = chain.predict(question=user_query).strip()

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

            return st.rerun()


def main():
    st.markdown("""
        <style>
            @keyframes gradient {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            body { direction: rtl !important; }
            .stApp {
                background: linear-gradient(-45deg, #FFFFFF, #F7F9FC, #F0F2F6, #F7F9FC);
                background-size: 400% 400%;
                animation: gradient 15s ease infinite;
            }
            [data-testid="stSidebar"] {
                background: rgba(255, 255, 255, 0.5) !important;
                backdrop-filter: blur(10px);
                border-right: 1px solid rgba(255, 255, 255, 0.2);
            }
            .block-container {
                max-width: 900px;
                padding-top: 2rem;
            }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            [data-testid="stChatMessage"] {
                animation: fadeIn 0.5s ease-in-out;
                border-radius: 20px;
                box-shadow: 0 2px 4px 0 rgba(0,0,0,0.05);
                padding: 14px 20px;
                margin-bottom: 12px;
                width: fit-content;
                max-width: 80%;
            }
            [data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                text-align: right; 
                float: right;
            }
            div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {
                background-color: #E6F2FF; 
                color: #1E1E1E;           
                float: left; 
            }
            .st-emotion-cache-1c7y2kd { width: 100%; }
        </style>
    """, unsafe_allow_html=True)
    initialize_session_state()
    render_sidebar()
    render_did_you_know_sidebar()

    if not st.session_state.get("kb_loaded"):
        render_welcome_screen()
    else:
        render_chat_interface()

if __name__ == '__main__':
    main()
