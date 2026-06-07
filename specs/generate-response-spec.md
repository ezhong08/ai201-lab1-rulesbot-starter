# Spec: `generate_response()`

**File:** `generator.py`
**Status:** Spec incomplete — fill in all blank fields before implementing

---

## Purpose

Given a user query and a list of retrieved rule chunks, generate a response that directly answers the question using only the retrieved text as context. The response must be grounded — it should not draw on the model's general knowledge of board games, only on what was retrieved.

---

## Input / Output Contract

**Inputs:**

| Parameter          | Type         | Description                                                                             |
| ------------------ | ------------ | --------------------------------------------------------------------------------------- |
| `query`            | `str`        | The user's original question                                                            |
| `retrieved_chunks` | `list[dict]` | Ranked list of chunks from `retrieve()`, each with `"text"`, `"game"`, and `"distance"` |

**Output:** `str`

A plain string containing the response to show the user. The response should:

- Answer the question using only the retrieved rule text
- Identify which game the answer comes from
- Acknowledge clearly when the answer is not found in the loaded rules

Returns a fallback string (not an error) when `retrieved_chunks` is empty.

---

## Design Decisions

_Complete the fields below before writing any code. Use your AI tool in Plan or Ask mode to help you reason through what belongs here — but the decisions are yours._

---

### Context formatting

_How will you format the retrieved chunks before passing them to the LLM? Describe the structure — not the code. Consider: will you label chunks by game? Include distance scores? Separate chunks with delimiters?_

```
I'll join the chunks into a single numbered context block, one entry per
chunk, each labeled with its source game so the model can both ground its
answer and cite the right game. Sketch of the structure:

    [1] (Catan) On your turn, roll both dice. All players collect resources...
    [2] (Catan) Build settlements at intersections of three hexes...
    [3] (Uno) The first player to reach 500 points across rounds wins...

Decisions:
- Label each chunk with its game in parentheses. The game name is the one
  piece of metadata we have, and the citation instruction depends on it being
  visible in the context.
- Number the chunks ([1], [2], ...). This gives clear delimiters between
  chunks (so the model doesn't blur two rules together) and a lightweight way
  for the model to refer to a specific passage.
- Do NOT include distance scores. They're an internal retrieval signal, not
  meaningful to the model's reasoning, and exposing a raw number invites the
  model to comment on it or treat it as confidence. Relevance handling stays
  in code, not in the prompt.
- Put each chunk on its own line/block so the boundaries are unambiguous.

The whole block is handed to the model as the only source of truth, preceded
by the grounding instruction below.
```

---

### System prompt — grounding instruction

_Write the exact system prompt instruction you will use to prevent the model from answering beyond the retrieved text. This is the most important design decision in this function._

```
You are RulesBot, an assistant that answers questions about board game rules.

Answer ONLY using the rule excerpts provided in the context below. These
excerpts are your single source of truth. Do not use any outside knowledge
about these games, even if you are confident you know the answer — your own
training knowledge is not allowed here.

- If the context contains the answer, respond using only what it says.
- If the context does NOT contain enough information to answer the question,
  do not guess or fill in gaps. Say clearly that the loaded rules don't cover
  it (see the fallback message).
- Never invent rules, numbers, or game names that are not in the context.

A confident wrong answer is worse than honestly saying you don't know.
At the end of every response, if the source was not mentioned yet, mention it in the form (source name)
```

---

### System prompt — citation instruction

_Write the exact instruction you will use to tell the model to identify which game its answer comes from._

```
State which game your answer comes from.

I fold this into the grounding system prompt as its own bullet rather than
making it a separate message. Because each context entry is already labeled
with its game in parentheses — e.g. "[1] (Catan) ..." — the model has the
game name right next to the text it's quoting, so naming the source is just a
matter of reading it off the label. When the answer draws on a single game,
the response should name that game (e.g. "According to the Catan rules...").
```

---

### Fallback behavior

_What should the response say when the answer isn't found in the loaded rule books? Write the exact fallback message._

```
There are two distinct "not found" situations, handled in two places:

1. retrieve() returned an EMPTY list (nothing in the vector store matched, or
   the collection is empty). This is a code-level fallback returned BEFORE we
   ever call the LLM — no point paying for a model call when there's no
   context. Exact message:

   "I couldn't find anything relevant in the loaded rule books. Try rephrasing
   your question — or check that your ingestion pipeline is working."

2. retrieve() returned chunks, but they don't actually answer the question
   (weak / off-topic matches). Here the LLM itself must decline, per the
   grounding instruction. It should respond with something like:

   "The loaded rules don't cover that — I couldn't find an answer to your
   question in the rule books I have."

   I keep this as a guideline rather than a rigid string because the model
   phrases it in context; the grounding prompt is what forces it to say so
   instead of guessing.
```

---

### Handling low-relevance chunks

_`retrieved_chunks` may include chunks with high distance scores (weak relevance). Will you filter these out before building context, pass them all in, or handle them another way? What are the tradeoffs?_

```
Decision: pass ALL the chunks in, no distance filtering in generator.py.
Lean on the grounding instruction to make the model ignore (and decline based
on) context that doesn't actually answer the question.

This is consistent with the matching decision in retrieve-spec.md: retrieve()
deliberately does no thresholding either, so relevance judgment lives in one
place — the LLM reading the context — rather than being split across a brittle
numeric cutoff in two functions.

Tradeoffs:
- Filter weak chunks here (e.g. drop distance > some cutoff):
  + Less noise reaches the prompt; slightly cheaper/shorter context.
  - The "right" cutoff is brittle and corpus-dependent, and with only 3
    chunks, filtering can leave the model with empty or near-empty context
    for questions it could otherwise have partially answered.
- Pass all in (chosen):
  + Simple, predictable; the model sees the best available evidence and the
    grounding prompt tells it to refuse if that evidence is insufficient.
  + A genuinely relevant chunk that happens to have a middling distance still
    gets a chance to be used.
  - A weak/off-topic chunk could mislead a poorly-grounded model — which is
    exactly why the grounding instruction is the most important design
    decision here.

The "distance" field is still present on every chunk, so a threshold can be
added later without changing the contract if testing shows it's needed.
```

---

### Message structure

_Describe how you will structure the messages list for the API call — what goes in the system message vs. the user message?_

```
Two messages: one system, one user.

  messages = [
      {"role": "system", "content": <grounding + citation instructions>},
      {"role": "user",   "content": "Context:\n<numbered block>\n\nQuestion: <query>"},
  ]

- System message = the standing behavior rules: who the bot is, the grounding
  instruction (answer only from context), the citation instruction (name the
  game), and what to do when the context is insufficient. These don't change
  from query to query, so they belong in the system role.
- User message = the per-request payload: the formatted context block plus the
  user's actual question. I put the context in the user message (not the
  system message) because it's data specific to THIS turn, and I label the two
  parts ("Context:" / "Question:") so the model can tell the retrieved
  evidence apart from what's being asked.

No prior conversation history is included — each call is stateless and answers
a single question against freshly retrieved context.
```

---

## Implementation Notes

_Fill this in after implementing and testing._

**Test query and response:**

```
Query: In Ticket to Ride, what do you do when you only have 1 Uno card left?
Response: The loaded rules don't cover what to do with an "Uno" card in Ticket to Ride. The rules for Uno cards are mentioned in the context for the game "Uno", not Ticket to Ride. Ticket to Ride rules mention destination tickets and train car pieces, but not Uno cards.
Correctly grounded? Yes, it acknowledges that it doesn't have the information instead of answering the question by taking outside knowledge connections between the two games.
Cited the right game? Yes, it noted both games that were relevant to the query.
```

**One thing you changed from your original spec after seeing the actual output:**

```
I noticed that the responses sometimes did not cite the source, so I modified it to explicitly mention the source if it hadn't already at the end, "At the end of every response, if the source was not mentioned yet, mention it in the form (source name)".
```
