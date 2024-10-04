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
    st.title("AI Powered Chatbot - Billing Team")
    st.subheader("Have fun using this...")
    st.write('Enter your software requirement(s) to generate test cases / code etc.')

    # Input a text box for user
    user_input = st.text_area("", height=150)

    # Button to send input to chatbot
    if st.button("Submit"):
        if user_input:
            with st.spinner('Generating...'):
                try:
                    # Call the chatbot function
                    response = chatbot(user_input)
                    st.write(f"Chatbot: {response}")
                except Exception as e:
                    st.error('An error occurred while generating test cases / code etc.')
                    st.error(e)
        else:
            st.write('Please enter a requirement to generate test cases / code etc.')
    st.write('Note* Kindly review these test cases and add/update based on your project specific requirements in detail.')

# Set up your OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Run the main function
if __name__ == "__main__":
    main()
