import streamlit as st
from rag_implement import conversation_chain 

st.set_page_config(page_title="AI Chatbot", page_icon="📄")
st.title("💬 Chat with your AI Chatbot")
with st.sidebar:
    st.header("Model Info")
    st.markdown("**Model:** `llama3.2`")
    st.markdown("**Mode:** Retrieval-Augmented Generation")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(message["user"])
    with st.chat_message("assistant"):
        st.markdown(message["bot"])

if prompt := st.chat_input("Ask something about your PDF..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation_chain.invoke({"question": prompt})
            answer = response["answer"]
            st.markdown(answer)

    # Save to session state
    st.session_state.chat_history.append({"user": prompt, "bot": answer})
