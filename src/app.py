import gradio as gr
from main import llm, create_faq_agent, Chatbot

faq_agent = create_faq_agent(llm)
chatbot = Chatbot(faq_agent)

def respond(message, history):
    answer = chatbot.chat(message, history=[])
    return answer

demo = gr.ChatInterface(
    fn=respond,
    title="🛡️ Insurance Claims Assistant",
    description="Ask me anything about insurance claims, coverage, deductibles, and premiums.",
    examples=[
        "What is a deductible?",
        "Does car insurance cover vandalism?",
        "How do I file a claim after an accident?",
        "What is collision vs comprehensive coverage?",
    ],
)

if __name__ == "__main__":
    demo.launch()