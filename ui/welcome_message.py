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
### Bienvenue sur DORA Assist

Hi! I’m Julia Bizeul’s AI agent, specializing in European legislation and regulations related to DORA**: my knowledge is based on
official European Union texts concerning the Digital Operational Resilience Act. 

**How to Use Me**
- When you ask a question, I’ll ensure its relevance by rephrasing it for you. To make your query as relevant as possible, you can view this rephrased query at any time by clicking the “Requête reformulée” option.
- Because technology is not meant to replace but to facilitate the lives of employees, I systematically and precisely cite my sources; you decide whether or not to display them using the “Sources citées” option.
- For this first version, I also want to provide you with the full context of this conversation so you can verify the relevance of my answers if they seem inaccurate to you. 
- Finally, my architecture is built on solid knowledge; I do not source information from the web, but exclusively from legal texts. These are all in English, but you can converse with me in any language. However, the sources I provide will be in English.
- This architecture is designed to be scalable, so in future versions, based on your feedback, I could easily integrate all available languages of official EU texts, as well as draft reports if needed, and provide a precise and well-founded analysis of the application of the directives outlined in the texts...

**Exemple de question :** « Que dit l’article 29 sur le risque de concentration des TIC ? »
""".strip()
