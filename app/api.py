"""
API HTTP FastAPI — même logique que l’UI : délègue à `app.main.answer_question`.

Lancer en local (depuis la racine du projet) :
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

Render : commande typique
    uvicorn app.api:app --host 0.0.0.0 --port $PORT

Variables d’environnement : identiques au reste du backend (ex. clés OpenAI, chemins
vers l’index — voir `app.config`).

Streamlit (Community Cloud) : définir `DORA_API_BASE_URL` sur l’URL publique du
service Render (sans slash final), ex. `https://dora-api.onrender.com`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.main import answer_question

app = FastAPI(title="DORA API", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_history: Optional[List[Dict[str, Any]]] = None
    return_context: bool = False


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    try:
        return answer_question(
            question=req.question.strip(),
            chat_history=req.chat_history,
            return_context=req.return_context,
        )
    except Exception as exc:  # noqa: BLE001 — exposé au client pour debug ops
        raise HTTPException(status_code=500, detail=str(exc)) from exc
