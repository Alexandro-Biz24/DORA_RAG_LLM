"""
Texte d’accueil affiché dans le fil de conversation (première visite + « Nouvelle conversation »).

Modifie uniquement la variable CONVERSATION_INTRO_MARKDOWN ci-dessous.
Tu peux utiliser la syntaxe Markdown Streamlit (titres, listes, gras, liens).
Ce message n’est pas envoyé au modèle : il sert uniquement à l’interface.
"""

# ---------------------------------------------------------------------------
# Écris ton introduction ici (projet DORA, objectifs, périmètre, bon usage…)
# ---------------------------------------------------------------------------

CONVERSATION_INTRO_MARKDOWN = """
### Bienvenue sur DORA

Je suis un **assistant de recherche juridique** : mes réponses s’appuient sur les documents
que vous avez indexés (RAG). Je cite les sources lorsque c’est pertinent.

**À propos de ce projet** — *remplacez ce paragraphe par votre propre texte* : décrivez ici
pourquoi DORA existe, à qui il s’adresse (équipe, contexte métier), et ce qu’il permet de faire
concrètement (ex. interroger le règlement DORA, préparer une analyse, etc.).

**Comment m’utiliser**
- Posez une question précise (article, thème, définition).
- Activez les options dans le menu de gauche si vous voulez voir les **sources** ou le **contexte** récupéré.

**Exemple de question :** « Que dit l’article 29 sur le risque de concentration des TIC ? »
""".strip()
