"""
Interface Streamlit — assistant juridique (RAG).
L’agent est invoqué via app.main.answer_question (même point d’entrée que le reste du projet),
sauf si la variable d’environnement DORA_API_BASE_URL est définie : dans ce cas les requêtes
passent par l’API FastAPI (backend hébergé, ex. Render).

Exécution locale (sans API) :
    streamlit run ui/streamlit_app.py

Avec backend distant :
    export DORA_API_BASE_URL=https://votre-service.onrender.com
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Racine du projet (parent du dossier ui/) — avant tout import `app` ou `ui.*`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_root_str = str(_PROJECT_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

import streamlit as st

from ui.welcome_message import CONVERSATION_INTRO_MARKDOWN


def _resolve_dora_api_base() -> str:
    """URL du backend FastAPI : env `DORA_API_BASE_URL` ou secret Streamlit du même nom."""
    env_val = (os.environ.get("DORA_API_BASE_URL") or "").strip()
    if env_val:
        return env_val.rstrip("/")
    try:
        sec = st.secrets.get("DORA_API_BASE_URL", "")
    except Exception:
        sec = ""
    return (str(sec).strip().rstrip("/") if sec else "")

# PDFs attendus : `<stem>.pdf` dans `data/pdf/`, avec `stem` = `doc_id` ou préfixe du `chunk_id` avant `__block`.
_PDF_DIR = _PROJECT_ROOT / "data" / "pdf"
# Limite d’aperçu inline (data URL) — au-delà : téléchargement seulement
_MAX_PDF_EMBED_BYTES = 6 * 1024 * 1024


def _resolve_logo_path() -> Optional[Path]:
    """Logo PwC : ui/asset/ ou ui/assets/ (selon l’emplacement du fichier)."""
    for rel in ("ui/asset/pwc_logo.png", "ui/assets/pwc_logo.png"):
        p = _PROJECT_ROOT / rel
        if p.is_file():
            return p
    return None


def _remote_answer_question(
    api_base: str,
    question: str,
    chat_history: Optional[List[Dict]] = None,
    return_context: bool = False,
) -> Dict[str, Any]:
    import httpx

    url = f"{api_base.rstrip('/')}/v1/chat"
    payload = {
        "question": question,
        "chat_history": chat_history or [],
        "return_context": return_context,
    }
    timeout = httpx.Timeout(600.0, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


@st.cache_resource
def _get_answer_fn(api_base: str):
    """
    Sans `api_base` : appelle le graphe en local (import paresseux).
    Avec `api_base` : POST vers `{api_base}/v1/chat` (backend FastAPI).
    """
    base = (api_base or "").strip().rstrip("/")
    if base:
        return lambda question, chat_history=None, return_context=False: _remote_answer_question(
            base,
            question,
            chat_history=chat_history,
            return_context=return_context,
        )
    from app.main import answer_question

    return answer_question


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _ensure_welcome_message() -> None:
    """Premier message assistant : intro projet (UI seulement, pas dans chat_history)."""
    if st.session_state.messages:
        return
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": CONVERSATION_INTRO_MARKDOWN,
            "is_welcome": True,
        }
    ]


def _src_value_present(val: Any) -> bool:
    """True si la valeur est exploitable (ignore None, chaîne vide, 'none')."""
    if val is None:
        return False
    s = str(val).strip()
    if not s or s.lower() == "none":
        return False
    return True


def _src_join_structure(label: str, nb: Any, name: Any) -> Optional[str]:
    """
    Ex. Titre + I + « Dispositions générales » → « Titre I — Dispositions générales ».
    Rien si numéro et nom sont tous deux absents.
    """
    has_nb = _src_value_present(nb)
    has_name = _src_value_present(name)
    if not has_nb and not has_name:
        return None
    nb_s = str(nb).strip() if has_nb else ""
    name_s = str(name).strip() if has_name else ""
    if has_nb and has_name:
        return f"{label} {nb_s} — {name_s}"
    if has_nb:
        return f"{label} {nb_s}"
    return f"{label} — {name_s}"


def _format_source_citation_md(src: Dict[str, Any]) -> str:
    """
    Citation lisible (hors style dictionnaire) :
    - bloc 1 : document, date, sous-titre (toujours affichés, avec repli si vide) ;
    - bloc 2 : titre / chapitre / section / article seulement si au moins une info utile.
    """
    doc = src.get("doc_title")
    date = src.get("doc_date")
    subtitle = src.get("doc_subtitle")

    doc_s = str(doc).strip() if _src_value_present(doc) else ""
    date_s = str(date).strip() if _src_value_present(date) else ""
    sub_s = str(subtitle).strip() if _src_value_present(subtitle) else ""

    # Ligne 1 : document + date (obligatoires visuellement ; repli discret)
    if doc_s and date_s:
        line_doc = f"**{doc_s}** · *{date_s}*"
    elif doc_s:
        line_doc = f"**{doc_s}**"
    elif date_s:
        line_doc = f"*{date_s}*"
    else:
        line_doc = "*Document non renseigné*"

    # Sous-titre : toujours une ligne dédiée (texte ou em dash si absent)
    line_sub = sub_s if sub_s else "—"

    structural: List[str] = []
    for label, nb_key, name_key in (
        ("Titre", "title_nb", "title_name"),
        ("Chapitre", "chapter_nb", "chapter_name"),
        ("Section", "section_nb", "section_name"),
        ("Article", "article_nb", "article_name"),
    ):
        part = _src_join_structure(label, src.get(nb_key), src.get(name_key))
        if part:
            structural.append(part)

    block = f"{line_doc}  \n*{line_sub}*"
    if structural:
        block += "  \n\n" + " · ".join(structural)
    return block


def _pdf_stem_from_chunk_id(chunk_id: Optional[str]) -> Optional[str]:
    """Ex. `CELEX_32022L2556_FR_TXT__block_2__chunk_0` → `CELEX_32022L2556_FR_TXT`."""
    if not chunk_id:
        return None
    s = str(chunk_id).strip()
    if "__block" in s:
        return s.split("__block", 1)[0].rstrip("_")
    return None


def _pdf_stem_from_source(src: Dict[str, Any]) -> Optional[str]:
    """Priorité au `doc_id` des métadonnées, sinon préfixe du `chunk_id`."""
    doc_id = src.get("doc_id")
    if _src_value_present(doc_id):
        return str(doc_id).strip()
    return _pdf_stem_from_chunk_id(src.get("chunk_id"))


def _norm_meta(val: Any) -> str:
    if not _src_value_present(val):
        return ""
    return str(val).strip()


def _block_index_from_chunk_id(chunk_id: Optional[str]) -> str:
    """Ex. `...__block_6__chunk_0` → `6`."""
    if not chunk_id:
        return ""
    m = re.search(r"__block_(\d+)__", str(chunk_id))
    return m.group(1) if m else ""


def _source_display_dedupe_key(src: Dict[str, Any]) -> tuple:
    """
    Une seule ligne d’affichage par « cible » citée : même PDF + même article
    (ou même section / chapitre / titre, ou même bloc si pas de structure).
    Les chunks multiples d’un même article ne sont pas répétés.
    """
    stem = (_pdf_stem_from_source(src) or "").strip()

    an = _norm_meta(src.get("article_nb"))
    am = _norm_meta(src.get("article_name"))
    if an or am:
        return ("article", stem, an, am)

    snb = _norm_meta(src.get("section_nb"))
    snm = _norm_meta(src.get("section_name"))
    if snb or snm:
        return ("section", stem, snb, snm)

    cnb = _norm_meta(src.get("chapter_nb"))
    cnm = _norm_meta(src.get("chapter_name"))
    if cnb or cnm:
        return ("chapter", stem, cnb, cnm)

    tnb = _norm_meta(src.get("title_nb"))
    tnm = _norm_meta(src.get("title_name"))
    if tnb or tnm:
        return ("title", stem, tnb, tnm)

    bid = _block_index_from_chunk_id(src.get("chunk_id"))
    if bid:
        return ("block", stem, bid)

    cid = _norm_meta(src.get("chunk_id"))
    return ("chunk", stem, cid)


def _dedupe_sources_for_display(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Conserve l’ordre d’origine, première occurrence gardée par clé de déduplication."""
    seen: set[tuple] = set()
    out: List[Dict[str, Any]] = []
    for src in sources:
        key = _source_display_dedupe_key(src)
        if key in seen:
            continue
        seen.add(key)
        out.append(src)
    return out


def _resolve_pdf_path(src: Dict[str, Any]) -> tuple[Optional[Path], Optional[str]]:
    """Retourne `(chemin si le fichier existe, stem attendu)` pour les messages d’erreur."""
    stem = _pdf_stem_from_source(src)
    if not stem:
        return None, None
    path = _PDF_DIR / f"{stem}.pdf"
    if path.is_file():
        return path, stem
    return None, stem


def _render_pdf_with_streamlit_pdf(path: Path, *, height: int, viewer_key: str) -> bool:
    """Visionneuse Streamlit `st.pdf` (paquet `streamlit-pdf` v1.x — la v2 casse avec Streamlit récent)."""
    try:
        st.pdf(str(path.resolve()), height=height, key=viewer_key)
        return True
    except Exception:
        return False


def _render_pdf_first_page_raster(path: Path) -> bool:
    """Repli : 1re page en image (PyMuPDF), si la visionneuse PDF n’est pas dispo."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False
    try:
        doc = fitz.open(path)
        if doc.page_count < 1:
            return False
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        st.image(io.BytesIO(pix.tobytes("png")), use_container_width=True)
        st.caption(
            f"Aperçu de la page 1 sur {doc.page_count}. "
            "Pour faire défiler tout le PDF dans l’app : `pip install streamlit-pdf` puis relancez."
        )
        doc.close()
        return True
    except Exception:
        return False


def _render_pdf_iframe_only(
    path: Path,
    *,
    iframe_height: int = 720,
    component_height: int = 740,
    viewer_key: str,
) -> None:
    """
    Affiche le PDF sans téléchargement.
    1) `st.pdf` + streamlit-pdf (recommandé, compatible fenêtre modale).
    2) Sinon 1re page en PNG via PyMuPDF.
    3) Dernier recours : iframe data-URL (souvent vide dans les iframes Streamlit — évité en pratique).
    """
    try:
        data = path.read_bytes()
    except OSError as e:
        st.error(f"Impossible de lire le fichier : {e}")
        return

    if len(data) > _MAX_PDF_EMBED_BYTES:
        st.warning(
            "Ce PDF est trop volumineux pour l’affichage intégré dans l’application. "
            "Utilisez **Télécharger** pour l’ouvrir sur votre poste."
        )
        return

    if _render_pdf_with_streamlit_pdf(path, height=iframe_height, viewer_key=viewer_key):
        return

    st.info(
        "Visionneuse PDF : installez une version compatible avec  "
        "`pip install \"streamlit-pdf>=1.0.8,<2\"`  puis redémarrez Streamlit "
        "(évitez la 2.x, elle peut provoquer une fenêtre vide)."
    )
    if _render_pdf_first_page_raster(path):
        return

    try:
        b64 = base64.standard_b64encode(data).decode("ascii")
        html = (
            f'<div style="height:{component_height}px;width:100%;overflow:auto;">'
            f'<embed src="data:application/pdf;base64,{b64}" '
            'type="application/pdf" width="100%" height="100%" '
            'style="min-height:680px;" />'
            f"</div>"
        )
        st.html(html, width="stretch")
    except Exception:
        st.error(
            "Impossible d’afficher le PDF ici. Utilisez **Télécharger** ou installez **streamlit-pdf**."
        )


def _render_pdf_download_only(path: Path, *, key: str) -> None:
    """Bouton de téléchargement uniquement."""
    try:
        data = path.read_bytes()
    except OSError as e:
        st.error(f"Impossible de lire le fichier : {e}")
        return

    st.download_button(
        "Télécharger le PDF",
        data=data,
        file_name=path.name,
        mime="application/pdf",
        key=key,
        use_container_width=True,
    )


@st.dialog("Lecture du document", width="large", dismissible=True, on_dismiss="rerun")
def _open_pdf_in_app_dialog(absolute_pdf_path: str) -> None:
    """Fenêtre modale pleine largeur : lecture dans l’app, sans téléchargement."""
    path = Path(absolute_pdf_path)
    if not path.is_file():
        st.error("Fichier introuvable.")
        return
    st.caption(path.name)
    viewer_key = f"pdfdlg_{hashlib.md5(str(path).encode('utf-8'), usedforsecurity=False).hexdigest()[:16]}"
    _render_pdf_iframe_only(
        path,
        iframe_height=780,
        component_height=800,
        viewer_key=viewer_key,
    )


def _render_sources(sources: List[Dict[str, Any]], *, widget_namespace: str) -> None:
    if not sources:
        st.info("Aucune source renvoyée.")
        return

    sources = _dedupe_sources_for_display(sources)

    ns = hashlib.md5(widget_namespace.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]

    for i, src in enumerate(sources):
        pdf_path, stem = _resolve_pdf_path(src)

        head_l, head_view, head_dl = st.columns([3, 1, 1], gap="small")
        with head_l:
            st.markdown(f"**Source {i + 1}**")
        with head_view:
            if pdf_path is not None:
                if st.button(
                    "Voir dans l’app",
                    key=f"{ns}_pdf_view_{i}",
                    type="primary",
                    use_container_width=True,
                    help="Ouvre le PDF dans une fenêtre pour le lire sans télécharger.",
                ):
                    _open_pdf_in_app_dialog(str(pdf_path.resolve()))
            else:
                st.caption("")
        with head_dl:
            if pdf_path is not None:
                dl_pop = st.popover(
                    "Télécharger",
                    key=f"{ns}_pdf_dl_pop_{i}",
                    type="secondary",
                    use_container_width=True,
                    on_change="rerun",
                    help="Enregistrer le PDF sur votre ordinateur.",
                )
                with dl_pop:
                    if dl_pop.open:
                        _render_pdf_download_only(
                            pdf_path,
                            key=f"{ns}_pdf_dl_{i}",
                        )
            elif stem:
                st.caption("PDF absent")
                st.caption(f"`{stem}.pdf`")
            else:
                st.caption("PDF indisponible")

        st.markdown(_format_source_citation_md(src))
        st.divider()


def _render_assistant_extras(
    msg: Dict[str, Any],
    *,
    show_rewritten: bool,
    show_sources: bool,
    show_context: bool,
    widget_namespace: str,
) -> None:
    if msg.get("is_welcome"):
        return
    if show_rewritten and msg.get("rewritten_query"):
        with st.expander("Requête reformulée"):
            st.code(msg["rewritten_query"])

    if show_sources and msg.get("sources"):
        with st.expander("Sources"):
            _render_sources(msg["sources"], widget_namespace=widget_namespace)

    if show_context and msg.get("context"):
        with st.expander("Contexte récupéré"):
            st.text(msg["context"])


def _inject_layout_css() -> None:
    """Fond dégradé (blanc → orange très clair) + bandeau latéral distinct du défilement principal."""
    st.markdown(
        """
        <style>
        /* Zone principale : dégradé léger type PwC */
        .stApp {
            background: linear-gradient(
                165deg,
                #ffffff 0%,
                #fffbf7 38%,
                #fff3e8 72%,
                #ffead9 100%
            ) !important;
        }
        /* Bloc principal (chat) : léger voile pour lisibilité */
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.45);
            border-radius: 14px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        /* Sidebar : même famille de teintes, reste dans son propre scroll */
        section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #ffffff 0%,
                #fff8f2 55%,
                #ffe8d6 100%
            ) !important;
            border-right: 1px solid rgba(255, 180, 120, 0.25);
        }
        section[data-testid="stSidebar"] > div {
            background: transparent !important;
        }
        [data-testid="stSidebarContent"] {
            max-height: 100vh;
            overflow-y: auto;
        }

        /* —— Options d’affichage : inspiration réglages iOS / Material (listes + interrupteurs) —— */
        .dora-sidebar-section-label {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 0.6875rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: rgba(100, 56, 32, 0.5);
            margin: 0.15rem 0 0.65rem 0.15rem;
            line-height: 1.35;
        }
        .dora-sidebar-section-hint {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 0.8125rem;
            color: rgba(90, 52, 30, 0.45);
            margin: -0.35rem 0 1rem 0.15rem;
            line-height: 1.45;
        }
        /* Cartes « ligne de réglage » autour de chaque interrupteur (clé Streamlit st-key-*) */
        section[data-testid="stSidebar"] div[class*="st-key-opt_sources"],
        section[data-testid="stSidebar"] div[class*="st-key-opt_context"],
        section[data-testid="stSidebar"] div[class*="st-key-opt_rewritten"] {
            background: rgba(255, 255, 255, 0.82) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(232, 119, 34, 0.12) !important;
            border-radius: 14px !important;
            padding: 0.35rem 0.5rem 0.35rem 0.65rem !important;
            margin-bottom: 0.5rem !important;
            box-shadow:
                0 1px 2px rgba(180, 90, 40, 0.04),
                0 4px 16px rgba(200, 100, 50, 0.06);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        section[data-testid="stSidebar"] div[class*="st-key-opt_sources"]:hover,
        section[data-testid="stSidebar"] div[class*="st-key-opt_context"]:hover,
        section[data-testid="stSidebar"] div[class*="st-key-opt_rewritten"]:hover {
            border-color: rgba(232, 119, 34, 0.22) !important;
            box-shadow:
                0 2px 4px rgba(180, 90, 40, 0.06),
                0 8px 22px rgba(200, 100, 50, 0.09);
        }
        /* Libellés des toggles : un peu plus nets */
        section[data-testid="stSidebar"] div[class*="st-key-opt_"] label p {
            font-weight: 600 !important;
            font-size: 0.9375rem !important;
            color: rgba(40, 28, 22, 0.92) !important;
        }
        /* Bouton nouvelle conversation : style pill / primaire doux */
        section[data-testid="stSidebar"] div[class*="st-key-dora_new_conversation"] button {
            border-radius: 9999px !important;
            font-weight: 600 !important;
            padding-top: 0.55rem !important;
            padding-bottom: 0.55rem !important;
            border: 1.5px solid rgba(232, 119, 34, 0.35) !important;
            background: linear-gradient(
                180deg,
                rgba(255, 255, 255, 0.98) 0%,
                rgba(255, 244, 232, 0.92) 100%
            ) !important;
            color: rgba(120, 55, 18, 0.95) !important;
            box-shadow: 0 2px 10px rgba(200, 95, 40, 0.12);
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }
        section[data-testid="stSidebar"] div[class*="st-key-dora_new_conversation"] button:hover {
            border-color: rgba(232, 119, 34, 0.55) !important;
            box-shadow: 0 4px 16px rgba(200, 95, 40, 0.18);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_display_toggles() -> tuple[bool, bool, bool]:
    """Interrupteurs pleine largeur + textes courts, détails en infobulle (pattern réglages système)."""
    st.markdown(
        '<p class="dora-sidebar-section-label">Affichage des réponses</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="dora-sidebar-section-hint">Activez ou désactivez les éléments sous chaque réponse de l’assistant.</p>',
        unsafe_allow_html=True,
    )

    show_sources = st.toggle(
        "Sources citées",
        value=True,
        key="opt_sources",
        help="Affiche la liste structurée des extraits juridiques utilisés (chunk, article, document, etc.).",
        width="stretch",
    )
    show_context = st.toggle(
        "Contexte récupéré",
        value=False,
        key="opt_context",
        help="Interroge le moteur avec le contexte brut et affiche le texte passé au modèle (plus lourd).",
        width="stretch",
    )
    show_rewritten = st.toggle(
        "Requête reformulée",
        value=True,
        key="opt_rewritten",
        help="Montre la requête de recherche dérivée de votre question et de l’historique de conversation.",
        width="stretch",
    )

    return show_sources, show_context, show_rewritten


# ---------------------------------------------------------------------------
# Config page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DORA — Assistant juridique",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_session()
_ensure_welcome_message()
_inject_layout_css()

answer_question = _get_answer_fn(_resolve_dora_api_base())

# ---------------------------------------------------------------------------
# Bandeau gauche (sidebar) : logo → titre → options (fixe vs défilement du chat)
# ---------------------------------------------------------------------------

with st.sidebar:
    logo_path = _resolve_logo_path()
    if logo_path is not None:
        st.image(str(logo_path), use_container_width=True)
    else:
        st.caption("Placez le fichier `pwc_logo.png` dans `ui/asset/` ou `ui/assets/`.")

    st.markdown("## DORA")
    st.caption("Assistant juridique (RAG)")

    st.divider()
    show_sources, show_context, show_rewritten = _render_sidebar_display_toggles()

    st.divider()
    if st.button(
        "Nouvelle conversation",
        use_container_width=True,
        type="secondary",
        key="dora_new_conversation",
    ):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Zone principale : conversation (défilement séparé)
# ---------------------------------------------------------------------------

st.title("Conversation")
st.caption(
    "Posez une question sur la réglementation, un article, un chapitre ou un thème. "
    "Les réponses sont ancrées sur les documents indexés."
)

for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_assistant_extras(
                msg,
                show_rewritten=show_rewritten,
                show_sources=show_sources,
                show_context=show_context,
                widget_namespace=f"m{msg_idx}",
            )

user_input = st.chat_input("Votre question juridique…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Recherche et rédaction de la réponse…"):
                result: Dict[str, Any] = answer_question(
                    question=user_input,
                    chat_history=st.session_state.chat_history,
                    return_context=show_context,
                )
        except Exception as exc:  # noqa: BLE001 — UI : message lisible
            st.error("Impossible d’obtenir une réponse pour le moment.")
            st.caption(str(exc))
            st.stop()

        answer_text: str = (result.get("answer") or "").strip() or "_(Réponse vide)_"
        st.markdown(answer_text)

        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": answer_text,
            "rewritten_query": result.get("rewritten_query"),
            "sources": result.get("sources"),
            "context": result.get("context") if show_context else None,
        }
        _render_assistant_extras(
            assistant_msg,
            show_rewritten=show_rewritten,
            show_sources=show_sources,
            show_context=show_context,
            widget_namespace="pending",
        )

    st.session_state.messages.append(assistant_msg)
    st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
    st.rerun()
