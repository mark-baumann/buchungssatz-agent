"""
Buchungssatz-Assistent — KI-gestützte Buchungssatz-Generierung aus Rechnungen.

- PDF-Upload mit automatischer Textextraktion
- LLM-Buchungssatz-Generierung nach SKR 03/04
- Demo-Modus mit Beispiel-Rechnung
- Human-in-the-Loop Feedback
- Eigener API-Key & Modell-Auswahl
"""
import os, tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv("/opt/data/finance-assistant/.env")
load_dotenv("/opt/data/.env")

from booking_engine import BookingEngine, BookingResult

st.set_page_config(
    page_title="Buchungssatz-Assistent",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .booking-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #e0e0e0; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
        font-family: 'Courier New', monospace; font-size: 1.1rem;
        border: 1px solid #2a2a4a;
    }
    .booking-card code {
        color: #e0e0e0 !important;
        background: rgba(0,0,0,0.35) !important;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .booking-card .soll { color: #ff6b6b; }
    .booking-card .haben { color: #51cf66; }
    .booking-card .an { color: #748ffc; }
    .metric-box { text-align: center; padding: 1rem; border-radius: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .metric-box.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .metric-box.amber { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); }
    .stButton > button { border-radius: 10px; font-weight: 600; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .konto-table { width: 100%; border-collapse: collapse; }
    .konto-table th { background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }
    .konto-table td { padding: 8px 12px; border-bottom: 1px solid #ddd; color: #1a1a2e; }
    .konto-table tr.soll-row { background: #fff5f5; }
    .konto-table tr.haben-row { background: #f0fff4; }
    @media (prefers-color-scheme: dark) {
        .konto-table td { color: #e0e0e0; }
        .konto-table tr.soll-row { background: #3a2020; }
        .konto-table tr.haben-row { background: #1a3a1a; }
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────
for key, default in {
    "invoice_text": None, "invoice_filename": None,
    "booking_result": None, "improved_result": None,
    "feedback_iteration": 0, "feedback_submitted": False,
    "history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Sidebar: Konfiguration ────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/accounting.png", width=64)
    st.markdown("## 🧾 Buchungssatz-KI")
    st.markdown("*Rechnungen hochladen — Buchungssätze automatisch generieren*")
    st.markdown("---")

    # ─── API-Key (versteckt) ────────────────────────────────────
    with st.expander("⚙️ Erweiterte Einstellungen", expanded=False):
        st.caption("API-Key & Modell — nur bei Bedarf ändern.")
        default_key = os.getenv("OPENAI_API_KEY", "")
        default_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        default_model = os.getenv("LLM_MODEL", "gpt-4o")

        user_key = st.text_input(
            "OpenAI API-Key",
            type="password",
            value="",
            placeholder="Dein eigener Key (leer = Server-Default)",
        )
        effective_key = user_key.strip() if user_key.strip() else default_key

        user_url = st.text_input(
            "API Base URL",
            value="",
            placeholder=f"Default: {default_url}",
        )
        effective_url = user_url.strip() if user_url.strip() else default_url

        model = st.selectbox(
            "🤖 KI-Modell",
            ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3-mini", "o4-mini"],
            index=0 if default_model == "gpt-4o" else 0,
        )

    st.markdown("---")

    if effective_key:
        st.success("✅ API bereit")
    else:
        st.error("❌ Kein API-Key")

    st.markdown("---")

    st.markdown("""
    ### 📋 So funktioniert's
    1. **Rechnung hochladen** (PDF)
    2. KI extrahiert alle Daten
    3. **Buchungssatz** wird generiert
    4. Mit **Feedback verbessern**

    ---
    ### 💡 Tipp
    Der Demo-Button lädt eine echte
    Pflegedienst-Rechnung.
    """)

    st.markdown("---")
    st.metric("Erstellte Buchungssätze", len(st.session_state.history))
    st.markdown("---")
    st.caption(f"Buchungssatz-Assistent v0.2 · {model}")


# ─── Engine (reaktiv) ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_engine_cached(key_hash: str, url_hash: str, model_hash: str):
    return BookingEngine(
        model=model,
        openai_api_key=effective_key,
        openai_base_url=effective_url,
    )

engine = _get_engine_cached(str(hash(effective_key)), str(hash(effective_url)), str(hash(model)))


# ─── Hilfsfunktionen ───────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    import fitz
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        doc = fitz.open(tmp_path)
        texts = []
        for page in doc:
            t = page.get_text()
            if t.strip():
                texts.append(t)
        return "\n\n".join(texts)
    finally:
        os.unlink(tmp_path)


def display_booking_result(result: BookingResult, iteration: int = 0):
    if iteration > 0:
        st.info(f"🔄 **Iteration {iteration}** — verbessert durch Ihr Feedback")

    # Buchungssatz — prominent im Dark-Mode-Style
    st.markdown("### 📝 Buchungssatz")
    # Parse Soll/Haben für farbige Darstellung
    bs = result.buchungssatz
    st.markdown(f"""
    <div class="booking-card">
        <span class="soll">📌 SOLL</span><br>
        <span class="an">an</span><br>
        <span class="haben">📌 HABEN</span><br><br>
        <code style="color:#e0e0e0;font-size:1.05rem;">{bs}</code>
    </div>
    """, unsafe_allow_html=True)

    # Erläuterung
    with st.expander("📖 Erläuterung", expanded=True):
        st.markdown(result.erlaeuterung)

    # Konten-Tabelle
    if result.konten:
        st.markdown("### 🗂️ Verwendete Konten")
        konten_data = []
        for k in result.konten:
            konten_data.append({
                "Konto": k.get("konto", ""),
                "Bezeichnung": k.get("bezeichnung", ""),
                "Soll (€)": f"{k.get('soll', 0):,.2f} €" if k.get("soll") else "—",
                "Haben (€)": f"{k.get('haben', 0):,.2f} €" if k.get("haben") else "—",
                "Typ": k.get("typ", ""),
            })
        st.dataframe(konten_data, use_container_width=True, hide_index=True)

    # Metriken
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-box green"><h2>{result.rechnungsbetrag:,.2f} €</h2><small>Rechnungsbetrag</small></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box amber"><h2>{result.buchungstyp or "—"}</h2><small>Buchungstyp</small></div>', unsafe_allow_html=True)

    if result.hinweise:
        st.info(f"💡 **Hinweis:** {result.hinweise}")


# ─── Hauptseite ────────────────────────────────────────────────────
st.markdown('<p class="main-header">🧾 Buchungssatz-Assistent</p>', unsafe_allow_html=True)
st.markdown("*Rechnung als PDF hochladen — die KI generiert automatisch den korrekten Buchungssatz nach SKR 03/04.*")

# Demo
DEMO_PDF = "/opt/data/20260717133302_Rechnung_Augustinum_Pflege.pdf"
has_demo = os.path.isfile(DEMO_PDF)

if has_demo and not st.session_state.invoice_text:
    with st.container(border=True):
        st.markdown("### 💡 Demo-Modus")
        st.markdown("Eine echte Pflegedienst-Rechnung (Augustinum → Techniker KK, 480,21 €) liegt bereit.")
        if st.button("🎯 Demo-Rechnung laden", type="secondary", use_container_width=True):
            with st.spinner("Lade..."):
                with open(DEMO_PDF, "rb") as f:
                    pdf_bytes = f.read()
                st.session_state.invoice_text = extract_text_from_pdf(pdf_bytes)
                st.session_state.invoice_filename = "Rechnung_Augustinum_Pflege.pdf"
            st.rerun()

st.markdown("---")

# Upload
uploaded = st.file_uploader("📎 Rechnung als PDF hochladen", type=["pdf"], key="invoice_upload")

if uploaded is not None and (st.session_state.invoice_filename != uploaded.name):
    with st.spinner("Extrahiere Text..."):
        st.session_state.invoice_text = extract_text_from_pdf(uploaded.getvalue())
        st.session_state.invoice_filename = uploaded.name
        st.session_state.booking_result = None
        st.session_state.improved_result = None
        st.session_state.feedback_iteration = 0
        st.session_state.feedback_submitted = False
    st.rerun()

# Rechnungstext
if st.session_state.invoice_text:
    st.success(f"✅ **{st.session_state.invoice_filename}** — {len(st.session_state.invoice_text):,} Zeichen")

    with st.expander("📄 Rechnungstext anzeigen", expanded=False):
        st.text(st.session_state.invoice_text)

    st.markdown("---")

    if not st.session_state.booking_result:
        st.markdown("### 🔍 Buchungssatz generieren")
        if st.button("🚀 Buchungssatz erstellen", type="primary", use_container_width=True):
            with st.spinner("🤖 Analysiere mit KI..."):
                result = engine.generate_booking(st.session_state.invoice_text)
                st.session_state.booking_result = result
                st.session_state.improved_result = None
                st.session_state.feedback_iteration = 0
                st.session_state.feedback_submitted = False
                st.session_state.history.append({
                    "filename": st.session_state.invoice_filename,
                    "betrag": result.rechnungsbetrag,
                    "buchungstyp": result.buchungstyp,
                })
            st.rerun()

    # Ergebnis
    current_result = st.session_state.improved_result or st.session_state.booking_result
    if current_result:
        st.markdown("---")
        st.markdown('<p class="main-header">📊 Ergebnis: Buchungssatz</p>', unsafe_allow_html=True)
        display_booking_result(current_result, st.session_state.feedback_iteration)

        st.markdown("---")

        if not st.session_state.feedback_submitted:
            st.subheader("💬 Human-in-the-Loop: Buchungssatz verbessern")
            with st.form(f"fb_{st.session_state.feedback_iteration}"):
                feedback_text = st.text_area(
                    "Was soll verbessert werden?",
                    placeholder="z.B.: 'Konto 4400 ist falsch — für Pflegeleistungen 4200 verwenden. Umsatzsteuer fehlt.'",
                    height=100,
                )
                rating = st.slider("Wie gut ist der Buchungssatz?", 1, 5, 3)
                c1, c2 = st.columns(2)
                with c1:
                    improve = st.form_submit_button("🔄 Verbessern", type="primary", use_container_width=True)
                with c2:
                    done = st.form_submit_button("✅ Korrekt", type="secondary", use_container_width=True)

                if improve and feedback_text.strip():
                    with st.spinner("Verbessere..."):
                        improved = engine.improve_with_feedback(
                            st.session_state.invoice_text, current_result, feedback_text,
                        )
                        st.session_state.improved_result = improved
                        st.session_state.feedback_iteration += 1
                    st.rerun()
                if done:
                    st.session_state.feedback_submitted = True
                    st.rerun()

        if st.session_state.feedback_submitted:
            st.success(f"✅ Buchungssatz finalisiert ({st.session_state.feedback_iteration} Iterationen).")
            if st.button("🔄 Neue Rechnung", type="secondary"):
                for k in ["invoice_text", "invoice_filename", "booking_result", "improved_result", "feedback_iteration", "feedback_submitted"]:
                    st.session_state[k] = None if k.endswith("result") or k.endswith("text") or k.endswith("filename") else (0 if k == "feedback_iteration" else False)
                st.rerun()

else:
    st.info("👆 Rechnung hochladen oder Demo-Button nutzen.")

st.markdown("---")
st.caption(f"Buchungssatz-Assistent · {model} · SKR 03/04 · Gesundheitswesen")
