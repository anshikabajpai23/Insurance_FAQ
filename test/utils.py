from langsmith import Client
from langchain_community.document_loaders import TextLoader
from openevals.llm import create_llm_as_judge
from openevals.prompts import RAG_RETRIEVAL_RELEVANCE_PROMPT, CORRECTNESS_PROMPT

client = Client()

def create_dataset_if_not_exists(dataset_name: str, examples: dict):
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(
            dataset_name=dataset_name, description="Insurance claims FAQ evaluation dataset."
        )
        client.create_examples(dataset_id=dataset.id, examples=examples)

def rag_relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    loader = TextLoader("knowledge_base/faq.md")
    docs = loader.load()
    evaluator = create_llm_as_judge(
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
        model="ollama:qwen2.5:7b",
        feedback_key="retrieval_relevance",
    )
    eval_result = evaluator(
        inputs=inputs,
        context=docs[0].page_content,
    )
    return eval_result

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="ollama:qwen2.5:7b",
        feedback_key="correctness",
    )
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs,
    )
    return eval_result

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