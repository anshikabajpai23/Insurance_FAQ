from dotenv import load_dotenv
import uuid
import faiss
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_ollama import OllamaEmbeddings  

load_dotenv(override=True)

from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b")
def create_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Get embedding size dynamically
    embedding_dim = len(embeddings.embed_query("hello world"))

    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    return vector_store

def create_faq_agent(llm: BaseChatModel):
    vector_store = create_vector_store()
    loader = DirectoryLoader("knowledge_base", glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(doc_splits)
    
    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(search_kwargs={"k": 2}),
        name="faq_assistant",
        description="Useful for answering questions about insurance claims, coverage, deductibles, premiums, and the claims process.",
    )
    
    prompt = """
    You're a helpful and concise assistant that helps users with insurance claims and coverage questions.

    <instructions>
    - MUST use word for word answers from the knowledge base, when possible.
    - If the answer is not in the knowledge base, say you don't know.
    - Never give legal or financial advice beyond what the FAQ states.
    </instructions>
    """
    return create_react_agent(
        llm,
        [retriever_tool],
        prompt=prompt,
        name="faq_agent",
    )
    
class Chatbot:
    def __init__(self, graph: StateGraph):
        self.graph = graph
        self.thread_id = str(uuid.uuid4())
        
    def chat(self, message, history = []):
        config = {"configurable": {"thread_id": self.thread_id}}
        state = {"messages": history + [{"role": "user", "content": message}]}
        result = self.graph.invoke(state, config)

        response = result['messages'][-1].content
        return response
        

if __name__ == "__main__":
    faq_agent = create_faq_agent(llm)
    chatbot = Chatbot(faq_agent)
    question = "How do I file an insurance claim after an accident?"
    print(f"Q: {question}")
    answer = chatbot.chat(message=question)
    print(f"A: {answer}")
