import streamlit as st
from main import query_answerer, eval_conversation, user_conversation, prompt1, prompt2

st.title("Code Writer :smile:")
language = st.selectbox("Select the language you want to write your code in ",
             ["Python", "Java", "C++", "JavaScript"])
prompt1 = prompt1.format(language = language)
prompt2 = prompt2.format(language = language)

query = st.text_input("### Ask any code to write :smile:")

button = st.button("Generate Answer")

if button:
    user_conversation.predict(input = prompt1)
    eval_conversation.predict(input = prompt2)
    with st.spinner("Generating the code......"):
        answer = query_answerer(query)
        st.code(answer, language = language.lower())
