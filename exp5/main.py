import streamlit as st
from transformers import pipeline
import wikipedia

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

st.set_page_config(page_title="AI in Education Tutor", layout="centered")
st.title("🎓 AI in Education: Virtual Tutor")

st.header("📘 Personalized Learning Suggestion")
st.write("Based on your last test score, we will recommend a topic for you to study.")

score = st.slider("Enter your last test score (0-100):", 0, 100, 80)

def recommend_topic(score):
    if score < 50:
        return "Recommended Topic: Basics of Python"
    elif score < 75:
        return "Recommended Topic: Functions and Loops"
    else:
        return "Recommended Topic: Object-Oriented Programming"

st.success(recommend_topic(score))

st.header("💡 Ask the AI Tutor a Question")

user_question = st.text_input("Type your question (e.g., What is Python?)")

if user_question:
    try:
        search_results = wikipedia.search(user_question)
        if not search_results:
            raise Exception("No relevant topic found on Wikipedia.")
        
        topic_page = search_results[2]
        summary = wikipedia.summary(topic_page, sentences=5)

        result = qa_pipeline(question=user_question, context=summary)

        st.markdown(f"**Tutor Answer:** {result['answer']}")
        st.caption(f"📚 Based on Wikipedia article: _{topic_page}_")
    except Exception as e:
        st.error(f"An error occurred: {e}")

