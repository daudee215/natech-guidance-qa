# Started as main.py from techwithtim/LocalAIAgentWithRAG
# (github.com/techwithtim/LocalAIAgentWithRAG), adapted in April 2026 to run
# over the Natech corpus. That repository carries no licence file, so this
# file is excluded from the MIT grant in LICENCE.
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about Natech risk management and Resilience of High-TECH industries and Critical infrastructures across Europe

Here are some relevant contents: {contents}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
while True: 
    print("\n\n------------------------------------")
    question = input("Ask your qeustion (q to quit): ")
    print("\n \n ")
    if question == "q":
        break
    contents = retriever.invoke(question)
    result = chain.invoke({"contents": contents, "question": question})

    print(result)