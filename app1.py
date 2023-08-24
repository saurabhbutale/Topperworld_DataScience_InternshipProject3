import streamlit as st
import pickle


# Load the trained model and vectorizer from pickle files
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to predict the news
def predict_fake_news(news_text):
    news_tfidf = vectorizer.transform([news_text])
    prediction = clf.predict(news_tfidf)
    return prediction[0]

# Streamlit app
def main():
    st.title('Fake News Detection')
    st.write("Enter the news text below to check if it's fake or real:")

    # Text input
    news_input = st.text_area("News Text", "")

    if st.button("Check"):
        if not news_input:
            st.warning("Please enter some news text.")
        else:
            prediction = predict_fake_news(news_input)
            if prediction == 'fake':
                st.error("This news is fake.")
            else:
                st.success("This news is real.")

if __name__ == '__main__':
    main()
