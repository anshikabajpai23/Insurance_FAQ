from langsmith import Client
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json
import re

client = Client()
judge_llm = ChatOllama(model="qwen2.5:7b")

def create_dataset_if_not_exists(dataset_name: str, examples: dict):
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(
            dataset_name=dataset_name, description="Insurance claims FAQ evaluation dataset."
        )
        client.create_examples(dataset_id=dataset.id, examples=examples)

# ── Parsers ────────────────────────────────────────────────────────────────────

def _parse_binary(content: str) -> dict:
    """For retrieval_relevance — stays 0 or 1."""
    match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {"score": int(data.get("score", 0)), "comment": str(data.get("comment", ""))}
        except (json.JSONDecodeError, ValueError):
            pass
    score = 1 if any(w in content.lower() for w in ["yes", "correct", "relevant", "accurate"]) else 0
    return {"score": score, "comment": content[:300]}

def _parse_ratio(content: str, numerator_key: str, denominator_key: str) -> dict:
    """
    Parses ratio score 0.0–1.0.
    Expects JSON like: {"present": 3, "total": 4, "comment": "..."}
    Score = numerator / denominator.
    Falls back to keyword binary if parsing fails.
    """
    match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            num = int(data.get(numerator_key, 0))
            den = int(data.get(denominator_key, 1))
            score = round(num / den, 2) if den > 0 else 0.0
            return {"score": score, "comment": str(data.get("comment", f"{num}/{den}"))}
        except (json.JSONDecodeError, ValueError, ZeroDivisionError):
            pass
    score = 1.0 if any(w in content.lower() for w in ["yes", "correct", "relevant", "accurate"]) else 0.0
    return {"score": score, "comment": content[:300]}

# ── Evaluators ─────────────────────────────────────────────────────────────────

def rag_relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Binary 0/1 — does the KB contain info to answer this question?"""
    loader = TextLoader("knowledge_base/faq.md")
    docs = loader.load()
    prompt = f"""Does the following knowledge base contain information relevant to answering the question?

Question: {inputs["question"]}

Knowledge Base:
{docs[0].page_content}

Respond with ONLY valid JSON: {{"score": 1, "comment": "reason"}} if relevant, {{"score": 0, "comment": "reason"}} if not."""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    result = _parse_binary(response.content)
    return {"key": "retrieval_relevance", "score": result["score"], "comment": result["comment"]}


def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """
    Ratio 0.0–1.0 — what fraction of reference facts appear in the agent answer?
    Score = present_facts / total_reference_facts
    Partial credit: 4/5 facts = 0.8, not 0.
    """
    prompt = f"""You are a strict insurance domain evaluator.

Step 1: List each distinct key fact from the Reference Answer, numbered.
Step 2: For each fact, write PRESENT if it appears in the Agent's Answer, or MISSING if not.
Step 3: Count them.

Question: {inputs["question"]}
Reference Answer: {reference_outputs["answer"]}
Agent's Answer: {outputs["answer"]}

Respond with ONLY valid JSON (no extra text):
{{"present": <integer>, "total": <integer>, "comment": "list any missing facts"}}"""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    result = _parse_ratio(response.content, "present", "total")
    return {"key": "correctness", "score": result["score"], "comment": result["comment"]}


def faithfulness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """
    Ratio 0.0–1.0 — what fraction of the agent's claims are grounded in the KB?
    Score = supported_claims / total_answer_claims
    Catches hallucinations — things the agent says that aren't in the FAQ at all.
    """
    loader = TextLoader("knowledge_base/faq.md")
    docs = loader.load()
    prompt = f"""You are evaluating whether an AI assistant's answer is grounded in the knowledge base (no hallucinations).

Step 1: List each distinct factual claim in the Agent's Answer, numbered.
Step 2: For each claim, write SUPPORTED if the knowledge base backs it up, or HALLUCINATED if it does not.
Step 3: Count them.

Knowledge Base:
{docs[0].page_content}

Agent's Answer:
{outputs["answer"]}

Respond with ONLY valid JSON (no extra text):
{{"supported": <integer>, "total": <integer>, "comment": "list any hallucinated claims, or none"}}"""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    result = _parse_ratio(response.content, "supported", "total")
    return {"key": "faithfulness", "score": result["score"], "comment": result["comment"]}


# ── Runner ─────────────────────────────────────────────────────────────────────

def evaluate(target, dataset_name: str, experiment_prefix: str, max_concurrency: int = 1):
    experiment_results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=[
            rag_relevance_evaluator,
            correctness_evaluator,
            faithfulness_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
    )
    return experiment_results