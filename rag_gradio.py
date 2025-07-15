import os
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import gradio as gr

# Config
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = OllamaLLM(model="mistral")  # Or use "llama3:8b-q4_K_M"

# Globals
vectorstore = None
qa_chain = None
chat_history = []
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_history = []  # Initialize globally

# Load PDF and create retriever
def process_pdf(file):
    global vectorstore, qa_chain, chat_history

    # Step 1: Load & split
    loader = PyPDFLoader(file.name)
    pages = loader.load()
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(pages)

    # Step 2: Embed & store
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # Step 3: Create retriever + chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define the prompt template
    system_prompt = """You are a helpful assistant. 
    Use the provided context to answer the question. 
    If the answer is not in the context, you can use prior knowledge and give a warning that says:
    "The answer is not in the context provided, but based on my prior knowledge, I can say: [your answer here]".

    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("Context: {context}\n\nQuestion: {question}")
    ])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    chat_history = []
    return "✅ PDF processed. You can now ask questions."



# Answer user queries
def ask_question(user_input):
    global vectorstore, qa_chain, chat_history

    if not vectorstore:
        return "⚠️ Please upload a PDF first.", []

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    relevant_docs = retriever.invoke(user_input)
    low_confidence = not relevant_docs or len(relevant_docs[0].page_content.strip()) < 30

    qa_inputs = {
        "question": user_input,
        "chat_history": [] if low_confidence else chat_history
    }

    response = qa_chain.invoke(qa_inputs)
    answer = response["answer"]

    if low_confidence:
        answer = "⚠️ I couldn’t find relevant info in the document.\n\n" + answer

    # Store in chat history as LIST of LISTS
    chat_history.append([user_input, answer])

    return "", chat_history


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Ask Your PDF (RAG with LangChain + Ollama)")

    file_input = gr.File(label="Upload your PDF", type="filepath")
    status_output = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(label="Document Chatbot")
    question_input = gr.Textbox(label="Ask a question", placeholder="What is this document about?")
    ask_button = gr.Button("Ask")

    file_input.change(process_pdf, inputs=[file_input], outputs=[status_output])
    ask_button.click(fn=ask_question, inputs=[question_input], outputs=[question_input, chatbot])


demo.launch()
