import sys
import os

# Ensure Python can find the 'app' module relative to the backend root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.retrieval import perform_search

# Load API key from environment
load_dotenv()

def get_llm():
    """
    Initializes OpenAI chat model (using gpt-4o-mini for speed and low cost).
    Temperature is set to 0.2 to balance factuality with a warm tone.
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

def stream_grounded_response(query: str):
    """
    Takes a user query, fetches context, and YIELDS a polite, friendly response token-by-token.
    """
    # 1. Fetch relevant chunks from ChromaDB
    relevant_chunks = perform_search(query)
    
    # Extract the text
    context_text = "\n\n".join([doc.page_content for doc in relevant_chunks])
    
    # Create citations and deduplicate them using set()
    citations = list(set([doc.metadata.get("source", "Unknown Document") for doc in relevant_chunks]))

    # YIELD 1: Send the citations immediately before the LLM starts typing
    yield {"type": "citations", "content": citations}

    # 2. Friendly System Prompt
    system_prompt = """You are a warm, highly empathetic, and proactive AI assistant for the University Registrar's Office. 
Your primary goal is to support students, ensure they feel heard, and give them comprehensive help.

Use the following pieces of retrieved context to answer the student's question. 
Here are your rules:
1. Empathy First: If a student asks a question related to security, health, well-being, or accommodations, adopt a highly supportive and reassuring tone. 
2. Be Proactive & Comprehensive (CRITICAL): Never just give the bare minimum answer. If a student asks for an email, and the context ALSO contains a phone number, office location, or related link for that department, you MUST provide all of them. Anticipate what else they might need!
3. Connect the Dots: If the exact answer isn't perfectly stated, share relevant clues from the context and explain how they might apply.
4. If the answer is truly nowhere to be found in the context, politely explain that you don't have that specific document yet, and suggest the best office to contact.

Context: 
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser() 

    # 3. Stream Response Token-by-Token
    for chunk in chain.stream({
        "context": context_text,
        "question": query
    }):
        # YIELD 2: Send each word/token as it is generated
        yield {"type": "token", "content": chunk}

if __name__ == "__main__":
    print("--- Testing OpenAI RAG Streaming Chain ---")
    valid_query = "What is the policy for exiting the degree after 3 years?"
    print(f"Question: {valid_query}")
    
    for item in stream_grounded_response(valid_query):
        if item["type"] == "citations":
            print(f"\nCitations: {item['content']}\nAnswer: ", end="")
        elif item["type"] == "token":
            # Print each token exactly as it arrives without line breaks
            print(item["content"], end="", flush=True)
    print("\n")