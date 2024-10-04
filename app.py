import streamlit as st
import openai

# Function to query GPT API
def chatbot(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response['choices'][0]['message']['content']

# Streamlit app starts here
def main():
    st.title("GPT Chatbot - Billing Team")
    st.write("Enter your question below:")

    # Input atext box for user
    user_input = st.text_area("How can I assist you today? ", height=150)

    # Button to send input to chatbot
    if st.button("Submit"):
        if user_input:
            # Call the chatbot function
            response = chatbot(user_input)
            st.write(f"Chatbot: {response}")
        else:
            st.write("Please enter a message.")

# Set up your OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Run the main function
if __name__ == "__main__":
    main()
