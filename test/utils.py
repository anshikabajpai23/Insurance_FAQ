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

def _parse(content: str) -> dict:
    match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {"score": int(data.get("score", 0)), "comment": str(data.get("comment", ""))}
        except (json.JSONDecodeError, ValueError):
            pass
    score = 1 if any(w in content.lower() for w in ["yes", "correct", "relevant", "accurate"]) else 0
    return {"score": score, "comment": content[:300]}

def rag_relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    loader = TextLoader("knowledge_base/faq.md")
    docs = loader.load()
    prompt = f"""Does the following knowledge base contain information relevant to answering the question?

Question: {inputs["question"]}

Knowledge Base:
{docs[0].page_content}

Respond with ONLY valid JSON: {{"score": 1, "comment": "reason"}} if relevant, {{"score": 0, "comment": "reason"}} if not."""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    result = _parse(response.content)
    return {"key": "retrieval_relevance", "score": result["score"], "comment": result["comment"]}

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    prompt = f"""You are a strict insurance domain evaluator.
Score 1 ONLY if the agent's answer contains ALL key facts from the reference answer.
Score 0 if the agent missed any key fact, added wrong information, or gave a vague answer.

Question: {inputs["question"]}
Reference Answer: {reference_outputs["answer"]}
Agent's Answer: {outputs["answer"]}

Respond with ONLY valid JSON: {{"score": 1, "comment": "reason"}} or {{"score": 0, "comment": "reason"}}"""

    response = judge_llm.invoke([HumanMessage(content=prompt)])
    result = _parse(response.content)
    return {"key": "correctness", "score": result["score"], "comment": result["comment"]}

def evaluate(target, dataset_name: str, experiment_prefix: str, max_concurrency: int = 1):
    experiment_results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=[
            rag_relevance_evaluator,
            correctness_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
    )
    return experiment_results