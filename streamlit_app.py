# Daniel C. P. Câmara
# AI app designed to help busy researchers go through their crescent number of unread papers.

# LIBRARIES AND SETUP
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain

import streamlit as st
import os
from tempfile import NamedTemporaryFile

# Função para carregar a chave da API e armazenar no session_state
def load_api_key():
    if "api_key" not in st.session_state:
        api_key = st.text_input("Insira sua chave da API OpenAI", type="password")
        if st.button("Salvar chave"):
            if api_key:
                st.session_state["api_key"] = api_key
                st.success("Chave da API salva!")
            else:
                st.error("Por favor, insira uma chave válida.")
    return st.session_state.get("api_key", None)

# 1.0 FUNÇÃO PARA CARREGAR E RESUMIR PDF
def load_summarize(file, api_key):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        file_path = tmp.name

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        model = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            api_key=api_key
        )
        
        prompt_template = """
        You are a helpful research assistant. You are specialized in public health, epidemiology of transmissible diseases, and epidemiology of non-transmissible diseases. Write a report from the following peer-reviewed scientific paper:
        {paper}
        
        Use the following Markdown format:
        # Abstract
        Begin the section with the full title of the paper followed by the full name of the first author. If there is more than one author, mention them as colleagues. Then, copy and paste the abstract without changing a single word.
        
        ## Introduction
        Use 3 to 7 numbered bullet points
        
        ## Methodology
        Describe the whole methodology section, including statistical analysis and statistical modelling if available. Use 3 to 10 numbered bullet points.
        
        ## Results
        Describe the main findings of the paper. Use 3 to 10 numbered bullets.
        
        ## Discussion and conclusions
        Read and summarize the most important points in the discussion section. Conclude with any future steps that the authors might discuss. Use 3 to 10 numbered bullets. 
        
        ## Most cited papers
        Finish with 1 to 5 numbered bullet points showing the most cited references in the paper.
        """
        
        prompt = PromptTemplate(input_variables=["paper"], template=prompt_template)
        llm_chain = LLMChain(prompt=prompt, llm=model)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="paper")
        response = stuff_chain.invoke(docs)

    finally:
        os.remove(file_path)

    return response["output_text"]

# 2.0 INTERFACE DO STREAMLIT
st.title("Busy Scientist App")
st.subheader("Carregue um documento PDF:")

# Carregar chave da API e salvar no session_state
api_key = load_api_key()

# Se a chave for carregada, rodar o restante do app
if api_key:
    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("Resumir o artigo"):
            with st.spinner("Resumindo..."):
                summary = load_summarize(uploaded_file, api_key)
                st.subheader("Resultados da Resumização:")
                st.markdown(summary)
    else:
        st.write("Nenhum arquivo carregado. Por favor, carregue um arquivo PDF.")
else:
    st.write("Por favor, insira sua chave da API para continuar.")
