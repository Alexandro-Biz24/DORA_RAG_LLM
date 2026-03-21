from typing import List, Dict, Optional

from app.graph import run_chat_graph


def answer_question(
    question: str,
    chat_history: Optional[List[Dict]] = None,
    return_context: bool = False
) -> Dict:
    return run_chat_graph(
        question=question,
        chat_history=chat_history,
        return_context=return_context
    )