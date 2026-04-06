import streamlit as st
import os
import re
from dotenv import load_dotenv
from utils.document_loader import load_all_categories
from utils.rag_pipeline import build_category_chains, query_rag
from utils.spell_check import correct_spelling, is_too_short_or_vague, needs_clarification
from utils.ragas_eval import evaluate_ragas

load_dotenv()

st.set_page_config(page_title="FinSight Pro", page_icon="💼", layout="wide")

st.markdown("""
<style>
    .chat-header {
        background: linear-gradient(135deg, #0f2942, #1e5799);
        padding: 1.2rem 2rem; border-radius: 14px; color: white;
        margin-bottom: 1.5rem; text-align: center;
    }
    .chat-header h2 { margin: 0; font-size: 1.7rem; }
    .chat-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.9rem; }
    .bubble-user {
        background: #1e5799; color: white;
        padding: 0.75rem 1.1rem; border-radius: 18px 18px 4px 18px;
        margin: 0.4rem 0 0.4rem auto; max-width: 70%; width: fit-content;
        font-size: 0.95rem; line-height: 1.5;
    }
    .bubble-bot {
        background: #f8fafc; color: #1e293b;
        padding: 0.85rem 1.1rem; border-radius: 18px 18px 18px 4px;
        margin: 0.4rem auto 0.4rem 0; max-width: 90%; width: fit-content;
        font-size: 0.95rem; line-height: 1.7; border: 1px solid #e2e8f0;
    }
    .bubble-warning {
        background: #fffbeb; color: #92400e;
        padding: 0.75rem 1.1rem; border-radius: 12px;
        margin: 0.4rem auto 0.4rem 0; max-width: 88%; width: fit-content;
        font-size: 0.93rem; border: 1.5px solid #fcd34d;
    }
    .category-badge {
        display: inline-block; font-size: 0.75rem; font-weight: 600;
        padding: 3px 10px; border-radius: 10px; margin-bottom: 6px;
    }
    .badge-health  { background: #dcfce7; color: #166534; }
    .badge-car     { background: #dbeafe; color: #1d4ed8; }
    .badge-banking { background: #fef9c3; color: #854d0e; }
    .badge-home    { background: #f3e8ff; color: #6d28d9; }
    .source-tag {
        display: inline-block; background: #eff6ff; color: #3b82f6;
        font-size: 0.72rem; padding: 2px 8px; border-radius: 10px;
        margin: 3px 2px 0; border: 1px solid #bfdbfe;
    }
    .ragas-bar {
        background: #f1f5f9; border-radius: 8px;
        padding: 6px 10px; margin-top: 8px; font-size: 0.75rem; color: #475569;
    }
    .metric-card {
        background: white; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1rem; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1e5799; }
    .metric-label { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
    div[data-testid="stButton"] button {
        border-radius: 20px !important; border: 1.5px solid #1e5799 !important;
        color: #1e5799 !important; background: white !important;
        font-size: 0.84rem !important; padding: 0.3rem 0.9rem !important; margin: 2px !important;
    }
    div[data-testid="stButton"] button:hover {
        background: #1e5799 !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="chat-header">
    <h2>💼 FinSight Pro</h2>
    <p>AI Financial Document Assistant · RAGAS Evaluated · Powered by Groq ⚡</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "retrievers": {}, "client": None, "docs_loaded": False,
    "messages": [], "category_names": {}, "active_category": None,
    "metrics": {
        "total_questions": 0, "found_count": 0, "not_found_count": 0,
        "response_times": [], "confidence_scores": [],
        "faithfulness_scores": [], "relevancy_scores": [], "precision_scores": [],
        "overall_ragas": [],
        "category_counts": {"health": 0, "car": 0, "banking": 0, "home": 0},
    }
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Auto load ─────────────────────────────────────────────────────────────────
def _get_main_buttons(retrievers=None):
    if retrievers is None:
        retrievers = st.session_state.retrievers
    loaded = list(retrievers.keys())
    btns = []
    if "health"  in loaded: btns.append("🏥 Health Insurance")
    if "car"     in loaded: btns.append("🚗 Car Insurance")
    if "banking" in loaded: btns.append("🏦 Banking & Loans")
    if "home"    in loaded: btns.append("🏠 Home Insurance")
    return btns

if not st.session_state.docs_loaded:
    data_dir = "data/"
    if os.path.exists(data_dir):
        with st.spinner("⏳ Loading documents by category..."):
            try:
                category_docs, category_names = load_all_categories(data_dir)
                if category_docs:
                    retrievers, client = build_category_chains(category_docs)
                    st.session_state.retrievers     = retrievers
                    st.session_state.client         = client
                    st.session_state.docs_loaded    = True
                    st.session_state.category_names = category_names
                    cat_buttons = _get_main_buttons(retrievers)
                    st.session_state.messages.append({
                        "role": "bot", "type": "text", "category": None,
                        "text": "Hi! 👋 Welcome to **FinSight Pro**.\n\nI can answer questions about your financial documents. Please select a category to get started.",
                        "sources": [], "buttons": cat_buttons
                    })
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "🏥 Health Insurance": "health",
    "🚗 Car Insurance":    "car",
    "🏦 Banking & Loans":  "banking",
    "🏠 Home Insurance":   "home",
}
CATEGORY_QUESTIONS = {
    "health":  ["Coverage details", "Maternity cover", "What is excluded?", "Annual benefit limits", "Dental cover", "Mental health cover"],
    "car":     ["What is covered?", "What is excluded?", "Accident benefits", "Windscreen cover", "No claims discount", "EU cover"],
    "banking": ["Loan eligibility", "Interest rates", "Repayment terms", "Maximum loan amount", "How to apply", "Early repayment penalty"],
    "home":    ["What is covered?", "What is excluded?", "Buildings cover", "Contents cover", "Claims process", "Policy limits"],
}
BADGE_INFO = {
    "health":  ("badge-health",  "🏥 Health Insurance"),
    "car":     ("badge-car",     "🚗 Car Insurance"),
    "banking": ("badge-banking", "🏦 Banking & Loans"),
    "home":    ("badge-home",    "🏠 Home Insurance"),
}
GREETINGS = ["hi", "hello", "hey", "hii", "good morning", "good afternoon", "good evening"]
AUTO_KEYWORDS = {
    "health":  ["health", "medical", "maternity", "dental", "hospital", "cigna", "cover", "clinical"],
    "car":     ["car", "motor", "vehicle", "allianz", "aviva", "driving", "windscreen", "accident"],
    "banking": ["loan", "bank", "kfw", "interest", "repayment", "borrow", "credit", "mortgage", "finance", "student loan"],
    "home":    ["home", "house", "property", "building", "contents", "flood", "subsidence"],
}


def _update_metrics(category, found, response_time, confidence, ragas=None):
    m = st.session_state.metrics
    m["total_questions"]    += 1
    m["response_times"].append(response_time)
    m["confidence_scores"].append(confidence)
    if found:
        m["found_count"]    += 1
    else:
        m["not_found_count"] += 1
    if category in m["category_counts"]:
        m["category_counts"][category] += 1
    if ragas:
        m["faithfulness_scores"].append(ragas["faithfulness"])
        m["relevancy_scores"].append(ragas["answer_relevancy"])
        m["precision_scores"].append(ragas["context_precision"])
        m["overall_ragas"].append(ragas["overall"])


def _avg(lst):
    return round(sum(lst) / len(lst), 1) if lst else 0.0


def handle_input(user_text: str):
    lower = user_text.lower().strip()

    # ── Greeting ──────────────────────────────────────────────────────────────
    if any(lower.startswith(g) for g in GREETINGS):
        st.session_state.messages.append({"role": "user", "text": user_text})
        st.session_state.active_category = None
        st.session_state.messages.append({
            "role": "bot", "type": "text", "category": None,
            "text": "Hi! 👋 How can I help you today? Please select a category.",
            "sources": [], "buttons": _get_main_buttons()
        })
        return

    # ── Too short or vague ────────────────────────────────────────────────────
    if is_too_short_or_vague(user_text):
        st.session_state.messages.append({"role": "user", "text": user_text})
        st.session_state.messages.append({
            "role": "bot", "type": "text",
            "text": "⚠️ Your question seems too short or vague. Could you please provide more details or rephrase your question?",
            "sources": [], "buttons": _get_main_buttons()
        })
        return

    # ── Category button ───────────────────────────────────────────────────────
    if user_text in CATEGORY_MAP:
        st.session_state.messages.append({"role": "user", "text": user_text})
        cat = CATEGORY_MAP[user_text]
        st.session_state.active_category = cat
        questions = CATEGORY_QUESTIONS.get(cat, [])
        label = user_text.replace("🏥","").replace("🚗","").replace("🏦","").replace("🏠","").strip()
        st.session_state.messages.append({
            "role": "bot", "type": "text", "category": cat,
            "text": f"Great! I'll search only the **{label}** documents.\n\nWhat would you like to know?",
            "sources": [], "buttons": questions[:4] + ["🔙 Back to categories"]
        })
        return

    # ── Back / navigation ─────────────────────────────────────────────────────
    if user_text == "🔙 Back to categories" or "main menu" in lower:
        st.session_state.messages.append({"role": "user", "text": user_text})
        st.session_state.active_category = None
        st.session_state.messages.append({
            "role": "bot", "type": "text", "category": None,
            "text": "What category can I help you with?",
            "sources": [], "buttons": _get_main_buttons()
        })
        return

    # ── Thanks / bye ──────────────────────────────────────────────────────────
    if any(w in lower for w in ["thank", "thanks", "bye", "goodbye"]):
        st.session_state.messages.append({"role": "user", "text": user_text})
        st.session_state.messages.append({
            "role": "bot", "type": "text", "category": None,
            "text": "You're welcome! 😊 Feel free to ask anything else.",
            "sources": [], "buttons": _get_main_buttons()
        })
        return

    # ── Spell check ───────────────────────────────────────────────────────────
    corrected, was_corrected = correct_spelling(user_text, st.session_state.client)

    # ── Needs clarification ───────────────────────────────────────────────────
    if needs_clarification(corrected):
        st.session_state.messages.append({"role": "user", "text": user_text})
        st.session_state.messages.append({
            "role": "bot", "type": "text",
            "text": "⚠️ I'm not sure I understood your question correctly. Could you please rephrase it or provide more details? For example: *'What does Cigna Gold cover for maternity?'*",
            "sources": [], "buttons": _get_main_buttons()
        })
        return

    # Show correction notice
    display_text = corrected
    correction_notice = ""
    if was_corrected:
        correction_notice = f"🔤 *Auto-corrected:* ~~{user_text}~~ → **{corrected}**\n\n"

    st.session_state.messages.append({"role": "user", "text": user_text})

    # ── Detect category ───────────────────────────────────────────────────────
    category = st.session_state.active_category
    if not category:
        for key, keywords in AUTO_KEYWORDS.items():
            if any(kw in corrected.lower() for kw in keywords):
                if key in st.session_state.retrievers:
                    category = key
                    break

    if not category:
        st.session_state.messages.append({
            "role": "bot", "type": "text",
            "text": "⚠️ Please select a category first so I can search the right documents for you.",
            "sources": [], "buttons": _get_main_buttons()
        })
        return

    # ── RAG query ─────────────────────────────────────────────────────────────
    try:
        answer, sources, found, response_time, confidence, context = query_rag(
            st.session_state.retrievers,
            st.session_state.client,
            corrected,
            category
        )

        # ── RAGAS evaluation ──────────────────────────────────────────────────
        ragas_scores = None
        if found and context:
            ragas_scores = evaluate_ragas(
                st.session_state.client,
                corrected, answer, context
            )

        _update_metrics(category, found, response_time, confidence, ragas_scores)

        questions = CATEGORY_QUESTIONS.get(category, [])

        if found:
            ragas_html = ""
            if ragas_scores:
                ragas_html = (
                    f"\n\n**📊 Answer Quality:** "
                    f"Faithfulness {ragas_scores['faithfulness']}% · "
                    f"Relevancy {ragas_scores['answer_relevancy']}% · "
                    f"Precision {ragas_scores['context_precision']}% · "
                    f"**Overall {ragas_scores['overall']}%**"
                )
            st.session_state.messages.append({
                "role": "bot", "type": "answer", "category": category,
                "text": correction_notice + answer + ragas_html,
                "sources": sources,
                "response_time": response_time,
                "confidence": confidence,
                "ragas": ragas_scores,
                "buttons": ["Ask another question", "🔙 Back to categories"]
            })
        else:
            # Try to generate a helpful summary from whatever was retrieved
            fallback_answer = ""
            if context and len(context.strip()) > 100:
                try:
                    fallback_response = st.session_state.client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are a financial document assistant. Based on the context provided, write a helpful paragraph summarising what IS available. Be honest about what is and isn't covered. Write in clear, plain English as a single paragraph."},
                            {"role": "user", "content": f"Context:\n{context}\n\nThe user asked about: {corrected}\n\nWrite a helpful summary paragraph of what the documents say about this topic."}
                        ],
                        temperature=0.1,
                        max_tokens=300
                    )
                    fallback_answer = fallback_response.choices[0].message.content.strip()
                except:
                    pass

            if fallback_answer and len(fallback_answer) > 30:
                st.session_state.messages.append({
                    "role": "bot", "type": "answer", "category": category,
                    "text": correction_notice + fallback_answer,
                    "sources": sources,
                    "response_time": response_time,
                    "confidence": confidence,
                    "ragas": None,
                    "buttons": ["Ask another question", "🔙 Back to categories"]
                })
            else:
                st.session_state.messages.append({
                    "role": "bot", "type": "text", "category": category,
                    "text": correction_notice + "We were unable to find a specific answer to your question in the current documents. This may be because the topic is not covered in detail, or the question needs more context. Please try asking a more specific question, such as mentioning the plan name or the specific benefit you are looking for.",
                    "sources": [],
                    "buttons": questions[:3] + ["🔙 Back to categories"]
                })

    except Exception as e:
        st.session_state.messages.append({
            "role": "bot", "type": "text", "category": None,
            "text": f"Something went wrong: {str(e)}",
            "sources": [], "buttons": _get_main_buttons()
        })


# ── Layout ────────────────────────────────────────────────────────────────────
chat_col, dash_col = st.columns([3, 1.2])

# ── METRICS DASHBOARD ─────────────────────────────────────────────────────────
with dash_col:
    st.markdown("### 📊 Live Metrics")
    m   = st.session_state.metrics
    total = m["total_questions"]
    found_rate = round((m["found_count"] / total) * 100) if total > 0 else 0

    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{total}</div>
        <div class="metric-label">Questions Asked</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#16a34a">{found_rate}%</div>
        <div class="metric-label">Answer Found Rate</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#d97706">{_avg(m['response_times'])}s</div>
        <div class="metric-label">Avg Response Time</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#7c3aed">{_avg(m['confidence_scores'])}%</div>
        <div class="metric-label">Retrieval Confidence</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # RAGAS scores
    st.markdown("**🎯 RAGAS Scores**")
    ragas_metrics = [
        ("Faithfulness",   m["faithfulness_scores"],  "#16a34a"),
        ("Ans. Relevancy", m["relevancy_scores"],      "#2563eb"),
        ("Ctx. Precision", m["precision_scores"],      "#7c3aed"),
        ("Overall",        m["overall_ragas"],         "#0f172a"),
    ]
    for label, scores, color in ragas_metrics:
        avg = _avg(scores)
        st.markdown(f"""<div class="metric-card" style="margin-bottom:4px">
            <div class="metric-value" style="color:{color};font-size:1.4rem">{avg}%</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Category usage
    if any(v > 0 for v in m["category_counts"].values()):
        st.markdown("**📂 By Category**")
        cat_labels = {"health": "🏥", "car": "🚗", "banking": "🏦", "home": "🏠"}
        for cat, count in m["category_counts"].items():
            if count > 0:
                pct = int((count / total) * 100) if total > 0 else 0
                st.markdown(f"{cat_labels[cat]} **{count}** ({pct}%)")

    # Response time chart
    if len(m["response_times"]) > 1:
        st.markdown("**⏱ Response Times**")
        import pandas as pd
        df = pd.DataFrame({"Q": list(range(1, len(m["response_times"]) + 1)), "s": m["response_times"]})
        st.line_chart(df.set_index("Q"), height=100)

# ── CHAT ──────────────────────────────────────────────────────────────────────
with chat_col:
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="bubble-user">{msg["text"]}</div>', unsafe_allow_html=True)
        else:
            text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', msg["text"])
            text = re.sub(r'~~(.*?)~~', r'<del>\1</del>', text)
            text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
            text = text.replace("\n", "<br>")

            cat = msg.get("category")
            badge_html = ""
            if cat and cat in BADGE_INFO:
                bc, bl = BADGE_INFO[cat]
                badge_html = f'<span class="category-badge {bc}">{bl}</span><br>'

            sources_html = "".join(
                f'<span class="source-tag">📄 {s}</span>'
                for s in msg.get("sources", [])
            )
            if sources_html:
                sources_html = f'<div style="margin-top:8px">{sources_html}</div>'

            meta_html = ""
            if msg.get("type") == "answer":
                rt = msg.get("response_time", 0)
                cf = msg.get("confidence", 0)
                meta_html = f'<div style="margin-top:6px;font-size:0.72rem;color:#94a3b8">⏱ {rt}s · 🎯 Retrieval: {cf}%</div>'

            bubble = "bubble-warning" if msg.get("type") == "warning" else "bubble-bot"
            st.markdown(f'<div class="{bubble}">{badge_html}{text}{sources_html}{meta_html}</div>', unsafe_allow_html=True)

            buttons = msg.get("buttons", [])
            if buttons and i == len(st.session_state.messages) - 1:
                cols = st.columns(min(len(buttons), 4))
                for j, btn in enumerate(buttons[:4]):
                    with cols[j]:
                        if st.button(btn, key=f"btn_{i}_{j}"):
                            if btn == "Ask another question":
                                cat2 = st.session_state.active_category
                                qs   = CATEGORY_QUESTIONS.get(cat2, [])
                                st.session_state.messages.append({
                                    "role": "bot", "type": "text", "category": cat2,
                                    "text": "What else would you like to know?",
                                    "sources": [], "buttons": qs[:4] + ["🔙 Back to categories"]
                                })
                                st.rerun()
                            else:
                                handle_input(btn)
                            st.rerun()

    user_input = st.chat_input("Type your question here...")
    if user_input:
        handle_input(user_input)
        st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💼 FinSight Pro")
    st.markdown("⚡ Groq · Llama 3.3 70B · RAGAS")
    st.markdown("---")
    st.markdown("### 📚 Documents")
    for cat, names in st.session_state.category_names.items():
        _, bl = BADGE_INFO.get(cat, ("", cat))
        st.markdown(f"**{bl}**")
        for name in names:
            st.markdown(f"📄 `{name}`")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_category = None
        st.session_state.metrics = {
            "total_questions": 0, "found_count": 0, "not_found_count": 0,
            "response_times": [], "confidence_scores": [],
            "faithfulness_scores": [], "relevancy_scores": [], "precision_scores": [],
            "overall_ragas": [],
            "category_counts": {"health": 0, "car": 0, "banking": 0, "home": 0},
        }
        st.session_state.messages.append({
            "role": "bot", "type": "text", "category": None,
            "text": "Chat cleared! Please select a category to get started.",
            "sources": [], "buttons": _get_main_buttons()
        })
        st.rerun()
