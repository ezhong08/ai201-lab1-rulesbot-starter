from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)


def generate_response(query, retrieved_chunks):
    """
    Generate a grounded answer from retrieved rule chunks.

    TODO — Milestone 3:

    `retrieved_chunks` is the list returned by retrieve(). Each item is a dict:
      - "text"     : the chunk text
      - "game"     : the game name
      - "distance" : similarity score (you can use this to filter weak matches)

    Before writing code, talk through these with your group:
      - How will you format the chunks into a context block for the prompt?
      - What instructions will stop the model from answering beyond what the
        rules say? (Grounding is the whole point — a confident wrong answer
        is worse than an honest "I don't know.")
      - How will you surface which game each answer comes from?

    Your response should:
      1. Answer using only the retrieved context — not the model's general knowledge
      2. Make clear which game the answer comes from
      3. Say so clearly when the answer isn't in the loaded rules

    Return the response as a plain string.
    """
    if not retrieved_chunks:
        return (
            "I couldn't find anything relevant in the loaded rule books. "
            "Try rephrasing your question — or check that your ingestion pipeline is working."
        )

    # Context formatting: a single numbered block, one entry per chunk, each
    # labeled with its source game. We keep "text" and "game" but drop
    # "distance" — that's an internal retrieval signal the model doesn't need.
    context = "\n".join(
        f"[{i}] ({chunk['game']}) {chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks, start=1)
    )

    # Grounding instruction: pin the model to the provided excerpts as its
    # only source of truth so it can't fall back on general knowledge.

    system_prompt = (
        "You are RulesBot, an assistant that answers questions about board game rules.\n\n"
        "Answer ONLY using the rule excerpts provided in the context below. These "
        "excerpts are your single source of truth. Do not use any outside knowledge "
        "about these games, even if you are confident you know the answer — your own "
        "training knowledge is not allowed here.\n\n"
        "- If the context contains the answer, respond using only what it says.\n"
        "- If the context does NOT contain enough information to answer the question, "
        "do not guess or fill in gaps. Say clearly that the loaded rules don't cover it.\n"
        "- Never invent rules, numbers, or game names that are not in the context.\n"
        "- State which game your answer comes from.\n\n"
        "A confident wrong answer is worse than honestly saying you don't know."
        "At the end of every response, if the source was not mentioned yet, mention it in the form (source name)"
    )

    # Message structure: instructions go in the system message; the retrieved
    # context and the user's question go in the user message.
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt}, # Same every time.
            {"role": "user", "content": user_message}, # Changes for each query.
        ],
    )

    return response.choices[0].message.content
