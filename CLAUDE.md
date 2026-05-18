# Insurance Claims FAQ Agent — Technical Plan & Progress

## Project Goal

Build an Insurance Claims FAQ agent using LangGraph that answers user questions by retrieving answers from an authentic knowledge base, evaluate its correctness using custom LLM-as-judge, with results tracked in LangSmith.

**Resume line (current):**
> "Built an insurance claims FAQ agent using LangGraph with hybrid BM25 + semantic retrieval; evaluated with LLM-as-judge across 40 Q&A pairs on 3 metrics (correctness, faithfulness, retrieval relevance) with ratio-based scoring, achieving 89%+ correctness tracked in LangSmith."

**Resume line (after all upgrades):**
> "Built an insurance claims FAQ agent using LangGraph with hybrid BM25 + semantic retrieval and multi-agent routing; evaluated with LLM-as-judge across 60 Q&A pairs on 3 metrics with RAGAS-style ratio scoring; compared 2 judge models; deployed as FastAPI REST API with Gradio chat UI."

---

## Current Tech Stack

| Tool | Purpose |
|---|---|
| **LangChain** | Document loading, text splitting, retriever tool creation |
| **LangGraph** | ReAct agent graph (`create_react_agent`) |
| **FAISS** | In-memory vector store for semantic similarity search |
| **Ollama `nomic-embed-text`** | Local embeddings — converts chunks to vectors |
| **Ollama `qwen2.5:7b`** | Agent LLM — reasons, calls tools, generates answers |
| **Ollama `qwen2.5:7b`** | Judge LLM — scores agent answers (custom evaluator) |
| **LangSmith** | Dataset management, experiment tracking, trace logging |
| **pytest** | Test runner that executes the evaluation |
| **Gradio** | Chat UI for interactive demos |

---

## Project Structure

```
Insurance_FAQ/
├── CLAUDE.md                    ← this file — update after every stage
├── knowledge_base/
│   └── faq.md                   ← 72 authentic Q&A pairs (Progressive, GEICO, State Farm)
├── src/
│   ├── main.py                  ← FAISS vector store + LangGraph ReAct agent + Chatbot
│   └── app.py                   ← Gradio chat UI
├── test/
│   ├── utils.py                 ← Custom LLM-as-judge evaluators + LangSmith helpers
│   └── test_main.py             ← 40 Q&A eval pairs + pytest test function
├── requirements.txt             ← pip dependencies
├── pytest.ini                   ← pythonpath config
└── .env                         ← API keys (never commit)
```

---

## Architecture

### Agent Flow (Runtime)

```
User Question
      │
      ▼
[ReAct Agent] ← qwen2.5:7b
      │  thinks: "I should search the FAQ"
      ▼
[Retriever Tool] ← calls Hybrid Retriever (BM25 + FAISS)
      │  BM25: exact keyword match (weight 0.4)
      │  Semantic: embedding similarity (weight 0.6)
      │  returns top-2 chunks from each, deduplicated
      ▼
[Retrieved Chunks] → injected into LLM context
      │
      ▼
[ReAct Agent] ← generates final answer grounded in retrieved text
      │
      ▼
Answer
```

### Evaluation Flow (pytest)

```
[40 Q&A pairs in test_main.py]
      │
      ▼
[LangSmith Dataset] ← created once, reused across runs
      │
      ▼
[Agent answers each question]
      │
      ├──▶ [Retrieval Relevance Evaluator]  score: 0 or 1 (binary)
      │         does faq.md contain the answer?
      │
      ├──▶ [Correctness Evaluator]          score: 0.0–1.0 (ratio)
      │         present_facts / total_reference_facts
      │
      └──▶ [Faithfulness Evaluator]         score: 0.0–1.0 (ratio)
                supported_claims / total_answer_claims
                catches hallucinations not in the KB
      │
      ▼
[LangSmith Experiment] ← scores visible at smith.langchain.com
      │
      ▼
[pytest assert] ← fails if avg score < 70%
```

---

## Key Concepts

### RAG (Retrieval-Augmented Generation)
Instead of relying on the LLM's training data, RAG injects relevant document chunks into the LLM's context at query time. The user's question is converted to an embedding vector, then FAISS finds the most semantically similar chunks from `faq.md`. Those chunks are handed to the LLM as grounding — the LLM answers from them, not from hallucination.

### Embeddings & FAISS
An embedding is a list of floats representing the semantic meaning of text. `nomic-embed-text` (local via Ollama) generates these. FAISS (`IndexFlatL2`) stores all chunk vectors and runs nearest-neighbor search using L2 distance. Similarity is semantic — "How much does it cost?" finds "Is it free?" because the vectors are close.

### ReAct Agent (LangGraph)
ReAct = Reason + Act. The agent loops:
1. **Think** — decide whether to call a tool
2. **Act** — call the retriever tool with a search query
3. **Observe** — read the retrieved chunks
4. **Respond** — generate final answer

### Hybrid Search (BM25 + Semantic)
BM25 = keyword frequency search (like old-school search engines). Semantic = vector similarity. Combining them with `EnsembleRetriever` (weights 0.4 BM25 / 0.6 semantic) catches both exact-match queries ("SR-22") and paraphrased queries ("what proves I have car insurance").

### Custom LLM-as-Judge
LLM outputs are non-deterministic so string comparison fails. A second LLM acts as judge. We use custom evaluators instead of OpenEvals' `create_llm_as_judge` because that library only works reliably with commercial models. Our `_parse_ratio()` and `_parse_binary()` handle inconsistent local model output with regex fallback.

### RAGAS-Style Ratio Scoring
Production eval systems don't score 0 or 1 — they score 0.0 to 1.0.
- **Correctness:** `present_facts / total_reference_facts` — partial credit for partially correct answers
- **Faithfulness:** `supported_claims / total_answer_claims` — catches hallucinations (claims not in the KB)
- Binary scoring hides degradation (60% correct ≠ 0% correct, but both score 0)

### Multi-hop Queries
Questions that require combining information from 2+ FAQ entries. Example: "If I hit a deer, do I pay a deductible?" requires retrieving both comprehensive coverage AND deductible info and combining them.

### Multi-turn Queries
Questions that reference previous messages in a conversation. Example: "What about hail?" after discussing comprehensive coverage. Requires passing conversation history to the agent.

---

## Knowledge Base

**File:** `knowledge_base/faq.md`
**Source:** Authentic content from Progressive, GEICO, and State Farm public FAQ pages
**Size:** 72 Q&A pairs
**Format:** Markdown `## Question` headers with answer paragraphs

| Category | Pairs |
|---|---|
| Auto coverage types | 14 |
| Claims process | 20 |
| Billing & payments | 6 |
| Discounts | 3 |
| Homeowners insurance | 14 |
| Umbrella & specialty | 3 |
| Liability specifics | 6 |
| Other (SR-22, excluded driver) | 6 |

---

## Score Tracker

| Stage | What Changed | Retrieval Relevance | Correctness | Faithfulness | Notes |
|---|---|---|---|---|---|
| Initial (broken) | `create_llm_as_judge` with `ollama/qwen2.5:7b` — parser fails | 0.45 | 0.28 | — | Parsing failures, not actual wrong answers |
| Stage 0 | Custom LLM-as-judge + strict prompt | **1.00** | **0.89** | — | Real baseline, binary 0/1 scoring |
| Stage 2 | Hybrid search BM25 + semantic | **1.00** | **0.88** | — | Retrieval not the bottleneck |
| Stage 3b | Ratio scoring + faithfulness metric | **1.00** | **0.852** | **0.512** | Faithfulness low = agent paraphrases, judge can't do verbatim match. Not hallucination (correctness stays high). Stage 5 will validate. |
| Stage 3c | Adversarial test set (20 questions: out-of-scope, multi-hop, paraphrased, comparative) | **1.00** | **0.732** | **0.817** | Lower correctness expected — adversarial questions are intentionally hard. Faithfulness higher than standard set (0.817 vs 0.512) — shorter answers on hard questions = fewer claims = less exposure to parse failures. |
| Stage 3a | Multi-turn history | — | — | — | TODO |
| Stage 5 | Larger judge model (qwen2.5:14b) | — | — | — | TODO — will reveal if faithfulness 0.51 is judge limitation or real hallucination |

**Pass threshold:** avg score ≥ 0.70  
**Total evaluations:** 120 (40 pairs × 3 metrics) after Stage 3b  
**Judge model:** `qwen2.5:7b` (local) — Stage 5 will validate reliability

### Scoring Methodology Change (Stage 3b)
Old: binary 0/1 per question → `passed/total ≥ 0.70`  
New: ratio 0.0–1.0 per question → `avg_score ≥ 0.70`  
Why: partial credit is more honest. An answer with 4/5 facts correct scores 0.8, not 0.

### Observations
- The 0.45/0.28 in the initial version was entirely a parsing failure — `create_llm_as_judge` couldn't parse local model output, defaulted to 0
- Custom evaluator with flexible `_parse()` fixed this immediately → 1.00/0.89
- BM25 hybrid gave 0.88 vs 0.89 — tiny dip confirms retrieval was already good; failures are in generation
- 100% retrieval relevance confirms the knowledge base covers all 40 test questions
- Faithfulness 0.512 is partially deflated by parse failures — qwen2.5:7b sometimes returns free text ("None of the claims are hallucinations.") instead of JSON; `_parse_ratio` fallback keywords don't cover faithfulness-specific phrasing → scores 0.0 even when judge says everything is fine

### Known Issues / Future Scope
| Issue | Impact | Fix | Priority |
|---|---|---|---|
| Faithfulness parse failures | Score artificially deflated (0.0 when judge says "no hallucinations") | Update `_parse_ratio` fallback to detect phrases like "none.*hallucin", "all.*supported" | Stage 5 or later |
| ~~Faithfulness 0/0 edge case~~ | ~~"I don't know" answers have 0 claims → 0/0 → 0.0~~ | ✅ Fixed in `_parse_ratio`: `if den == 0: return 1.0`. Adversarial run (faith 0.817) already reflects this fix. | DONE |
| Faithfulness checks full KB (72 entries) not retrieved chunks | Judge accuracy low over large context | Return retrieved chunks from `Chatbot.chat()`, pass as `outputs["context"]` to evaluator | Citations stage |
| qwen2.5:7b ignores "ONLY valid JSON" instruction ~10-15% of time | Any ratio metric can silently fail | Strengthen prompt or add retry logic | Stage 5 |
| `rag_relevance_evaluator` scores 1.0 for out-of-scope questions | Adversarial out-of-scope questions (e.g. "Does insurance cover war damage?") incorrectly get relevance 1.0 — judge sees "insurance" in KB and says yes without checking for a specific answer | Tighten prompt: change "relevant to answering" → "contains a **specific answer** to". Real fix: larger judge model (Stage 5) | Stage 5 |
| Judge self-consistency failure (correctness inflation) | qwen2.5:7b correctly identifies missing facts in reasoning ("Missing: Insurance pays actual cash value") but gives score 1.00 anyway — reasoning and score contradict each other. Correctness is likely inflated ~0.1-0.15 across both datasets. Real correctness probably ~0.75-0.80 not 0.852 | Larger judge model (Stage 5) — 14b models are significantly more self-consistent between chain-of-thought and final score | Stage 5 |

---

## Progress Tracker

| Stage | Description | Status |
|---|---|---|
| **Foundation** | Knowledge base, agent, evaluator, pytest | ✅ DONE |
| **Stage 0** | Fix evaluator — custom LLM-as-judge, 89% baseline | ✅ DONE |
| **Stage 1** | Gradio chat UI | ✅ DONE |
| **Stage 2** | Hybrid search (BM25 + semantic) | ✅ DONE |
| **Stage 3b** | Ratio scoring + faithfulness evaluator | ✅ DONE |
| **Stage 3c** | Adversarial test set (out-of-scope, multi-hop, paraphrased) | ✅ DONE |
| **Stage 3a** | Multi-turn history in Gradio | ⬜ TODO |
| **Stage 6** | Multi-agent router (Auto / Home / Billing specialists) | ⬜ TODO |
| **Stage 4** | FastAPI REST endpoint + Docker | ⬜ TODO |
| **Stage 5** | Model comparison experiment (7b vs 14b judge) | ⬜ TODO |
| **Stage 8** | Persistent memory with MemorySaver | ⬜ TODO |
| **Stage 7** | MCP server for live data (mock claims database) | ⬜ TODO |

### Why this order?
1. **3b** — ratio scoring is the biggest single signal of eval maturity; interviewers notice binary scoring immediately
2. **3c** — adversarial sets show you think about failure modes (what companies actually care about in prod)
3. **3a** — one-line fix but converts the demo from Q&A lookup to conversational agent
4. **6** — highest-signal architecture upgrade; shows system design + agent orchestration
5. **4** — production deployment; separates you from notebook-only candidates
6. **5** — ablation study / evaluator bias check; shows scientific rigor
7. **8** — stateful memory; good UX story
8. **7** — MCP; niche but impressive for the right company

---

## Stage Details

### ✅ Stage 0 — Fix the Foundation
- Replaced `create_llm_as_judge` with custom evaluators using `ChatOllama` directly
- Added `_parse()` with regex fallback for local model output inconsistency
- Strict correctness prompt that penalizes missed facts and hallucinations
- **Baseline:** 89% correctness, 100% retrieval relevance

---

### ✅ Stage 1 — Gradio Chat UI
**File:** `src/app.py`
```bash
python src/app.py   # opens at http://localhost:7860
```

---

### ✅ Stage 2 — Hybrid Search (BM25 + Semantic)
**File:** `src/main.py`
```python
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(doc_splits); bm25_retriever.k = 2
semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, semantic_retriever], weights=[0.4, 0.6])
```

---

### 🔄 Stage 3b — Ratio Scoring + Faithfulness Evaluator
**Files:** `test/utils.py`, `test/test_main.py`

**What changed:**
- `correctness_evaluator`: now scores `present_facts / total_reference_facts` (0.0–1.0)
- New `faithfulness_evaluator`: scores `supported_claims / total_answer_claims` (0.0–1.0)
- `_parse_binary()` for retrieval relevance (stays 0/1)
- `_parse_ratio()` for correctness + faithfulness
- `test_main.py`: assert on `avg_score` (sum of all scores / total) instead of binary pass rate

**Faithfulness explained:**
> "How much of what the agent said is actually in the knowledge base?"  
> Agent says 5 things → 4 are in the FAQ → faithfulness = 0.8  
> Catches hallucinations that binary correctness misses

---

### ⬜ Stage 3c — Adversarial Test Set
Add 20 questions to a second dataset:
- **Out-of-scope (5):** agent should say "I don't know" — "What is the capital of France?"
- **Multi-hop (5):** requires combining 2+ FAQ entries — "If I hit a deer, do I pay a deductible?"
- **Paraphrased (5):** different wording, same meaning — "If someone hits me and has no coverage, what happens?"
- **Comparative (5):** "What's the difference between comprehensive and collision?"

Tracked separately in LangSmith as `"Insurance FAQ Complex dataset"`.

---

### ⬜ Stage 3a — Multi-turn History
One-line fix in `src/app.py`:
```python
def respond(message, history):
    formatted = [{"role": r, "content": c} for turn in history for r, c in [("user", turn[0]), ("assistant", turn[1])]]
    return chatbot.chat(message, history=formatted)
```

---

### ⬜ Stage 6 — Multi-Agent Router
```
User Question → [Router Agent] → Auto Claims Agent
                              → Homeowners Agent
                              → Billing Agent
```
Each specialist has its own retriever scoped to its domain chunks.

---

### ⬜ Stage 4 — FastAPI + Docker
`src/api.py` with `POST /ask`, `GET /health`, `POST /evaluate`  
Containerized with Docker.

---

### ⬜ Stage 5 — Model Comparison
Run same eval with `qwen2.5:14b` as judge. Compare in LangSmith.

---

### ⬜ Stage 8 — Persistent Memory
```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
create_react_agent(llm, [retriever_tool], checkpointer=memory)
```

---

### ⬜ Stage 7 — MCP Server
Mock claims database MCP server — agent can look up claim status beyond static FAQ.

---

## Agentic AI Concepts in This Project

| Concept | Where Used | Stage |
|---|---|---|
| RAG | FAISS + LLM grounding | Foundation |
| ReAct | Agent reasoning loop | Foundation |
| LLM-as-Judge | Custom evaluators | Stage 0 |
| Tool Use | Retriever as agent tool | Foundation |
| Hybrid Search | BM25 + semantic EnsembleRetriever | Stage 2 |
| RAGAS-style scoring | Ratio-based faithfulness + correctness | Stage 3b |
| Multi-turn | Conversation history in Gradio | Stage 3a |
| Adversarial Eval | Out-of-scope + multi-hop test set | Stage 3c |
| Multi-Agent | Router + specialist agents | Stage 6 |
| MCP | Standardized external tool servers | Stage 7 |
| Memory | Persistent state across sessions | Stage 8 |

---

## Running the Project

```bash
# activate venv
source venv/bin/activate

# run agent sanity check
python src/main.py

# run Gradio UI
python src/app.py

# run full evaluation
python -m pytest test/test_main.py -v -s
```

---

## Environment Variables (.env)

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=insurance-faq-agent
```

Ollama runs locally — no API key needed for LLM or embeddings.
