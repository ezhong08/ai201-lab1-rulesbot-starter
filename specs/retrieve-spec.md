# Spec: `retrieve()`

**File:** `retriever.py`
**Status:** Specs should be complete

---

## Purpose

Given a user's natural language query, find the most relevant chunks from the vector store using semantic similarity search. Return them ranked by relevance so that `generate_response()` can use them as context.

---

## Input / Output Contract

**Inputs:**

| Parameter   | Type  | Description                                                                |
| ----------- | ----- | -------------------------------------------------------------------------- |
| `query`     | `str` | The user's natural language question                                       |
| `n_results` | `int` | Maximum number of chunks to return (default: `N_RESULTS` from `config.py`) |

**Output:** `list[dict]`

Each dict in the returned list must contain exactly these keys:

| Key          | Type    | Description                                                   |
| ------------ | ------- | ------------------------------------------------------------- |
| `"text"`     | `str`   | The chunk text                                                |
| `"game"`     | `str`   | The game name this chunk came from                            |
| `"distance"` | `float` | Cosine distance score — lower means more similar to the query |

Results should be ordered from most to least relevant (lowest to highest distance). Returns an empty list `[]` if the collection contains no documents.

---

## Design Decisions

_Complete the fields below before writing any code. Use your AI tool in Plan or Ask mode to help you reason through what belongs here — but the decisions are yours._

---

### Query approach

_Describe how you will use `_collection.query()` to find relevant chunks. What arguments will you pass, and why?_

```
results = _collection.query(
    query_texts=[query],
    n_results=n_results,
    include=["documents", "metadatas", "distances"],
)

- query_texts=[query] : Chroma embeds the raw query string with the same
  sentence-transformers model used at ingestion, then runs the similarity
  search. It takes a LIST because Chroma can batch many queries at once; we
  only have one, so we wrap our single query in a one-element list.
- n_results : caps how many chunks come back (defaults to N_RESULTS = 3).
- include=["documents", "metadatas", "distances"] : this is exactly the data
  the output contract needs — documents -> "text", metadatas -> "game",
  distances -> "distance". I deliberately do NOT include "ids" or
  "embeddings"; the contract doesn't use them, so there's no reason to pull
  them back.
```

---

### Return structure

_Sketch out what one item in your return list looks like as a concrete example. Where does each field come from in the query results?_

```
One item, for the i-th result:

{
    "text":     "At the start of your turn, roll both dice....",
    "game":     "Catan",
    "distance": 0.41,
}

Field sources (after unwrapping the per-query nesting with [0]):
  "text"     <- results["documents"][0][i]
  "game"     <- results["metadatas"][0][i]["game"]   (the only metadata key
                we store at ingestion, see embed_and_store in retriever.py)
  "distance" <- results["distances"][0][i]

Build the full list by zipping these three parallel lists together, e.g.:

    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [
        {"text": t, "game": m["game"], "distance": d}
        for t, m, d in zip(docs, metas, dists)
    ]

Chroma already returns them ordered by ascending distance (most relevant
first), so the contract's "most to least relevant" ordering comes for free.
```

---

### Handling the nested result structure

_`_collection.query()` returns nested lists. Describe what index you need to access to get the actual list of results for a single query, and why the nesting exists._

```
I need index [0].

query() returns a dict where each value is a list-of-lists, e.g.
results["documents"] looks like:

    [ ["chunk for query 0", "chunk for query 0", ...] ]
      ^------- one inner list PER query in query_texts -------^

The OUTER list has one element per query you passed in query_texts. Since I
pass a single query, query_texts=[query], the outer list has exactly one
element, and results["documents"][0] is the actual list of matching chunks
for my query. Same pattern for ["metadatas"][0] and ["distances"][0].

The nesting exists because query() is designed for BATCHED search — you can
ask about many queries in one call, and each gets its own inner list of
results. We just happen to use the batch-of-one case.
```

---

### Relevance threshold

_Will you filter out results above a certain distance score, or return all `n_results` regardless of how relevant they are? What are the tradeoffs of each approach?_

```
Decision: return all n_results, with NO hard distance threshold in retrieve().
Push the "is this actually relevant?" judgment downstream to
generate_response(), which is already designed to ground its answer in the
context (or honestly say "I don't know") based on what it receives.

Tradeoffs:

- Hard threshold (e.g. drop anything with distance > 0.8):
  + Stops obviously-irrelevant junk from reaching the LLM.
  - The "right" cutoff is brittle: cosine distances vary a lot by query and
    corpus, and a fixed number that works for one question may wrongly return
    [] for a perfectly valid one. With only 3 results, over-filtering is a
    real risk, and config.py has no threshold constant to tune.

- Return all n_results (chosen):
  + Simple, predictable, and keeps retrieval's single job: "find the closest
    chunks." The LLM is better at deciding relevance in context than a raw
    distance number is.
  + retrieve() stays a clean, easily testable building block.
  - Weak matches can reach the prompt; we rely on generator.py's grounding
    instructions to refuse rather than hallucinate.

If filtering proves necessary later, the "distance" field is in every result,
so a threshold can be added without changing the contract.
```

---

### Edge cases

_How does your implementation behave when: (a) the collection is empty, (b) the query matches no chunks well, (c) the query matches chunks from multiple games?_

```
(a) Empty collection: handled before querying. The existing guard
    `if _collection.count() == 0: return []` short-circuits, so we never call
    query() on an empty store. Satisfies the contract's "returns [] if the
    collection contains no documents." generate_response() already turns []
    into a friendly "couldn't find anything" message.

(b) Query matches nothing well: query() still returns the n closest chunks,
    just with HIGH distance scores — it never returns fewer than n_results
    (unless the whole collection is smaller than n_results). Per the threshold
    decision above, we still return them; grounding at generation time is
    responsible for declining to answer when the context doesn't actually
    cover the question.

(c) Query spans multiple games: totally fine. Results are ranked purely by
    semantic distance, so the list can mix chunks from different games
    (e.g. two Catan chunks and one Codenames chunk). Each dict carries its own
    "game" field, so the answer layer can attribute each piece of context to
    the right game instead of blending rules from different games together.
```

---

## Implementation Notes

_Fill this in after implementing, before moving to Milestone 3._

**Test query and top result returned:**

```
Query: How do you set up the board in Catan?
Top result game: Catan
Distance score: 0.380
Does it make sense? Yes it makes sense because the question was about Catan.
```

**One thing about the query results that surprised you:**

```
The top result for the Catan query was about an overview of Catan. Words like "Catan" and "board" matched, but the top result did not seem to answer the question very well.
```
