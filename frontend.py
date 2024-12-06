import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from utils import ImageCaptionTool, ObjectDetectionTool
from dotenv import load_dotenv
import os

##############################
### SET THE OPENAI API KEY ###
##############################

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

##############################
###### Initialize Agent ######
##############################

tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key=api_key,
    temperature=0.7,
    model_name="gpt-3.5-turbo"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

##############################
###### Helper Function #######
##############################

def extract_image_context(image):
    """
    Extract caption and detected objects from the image.
    """
    caption_tool = ImageCaptionTool()
    object_tool = ObjectDetectionTool()

    # Generate caption
    caption = caption_tool._run(image)

    # Detect objects
    detected_objects = object_tool._run(image)

    # Combine and format results
    return f"Caption: {caption}\nObjects detected:\n{detected_objects}"

##############################
### Streamlit Chat Layout ####
##############################

image_url = "https://i.pinimg.com/originals/f8/44/8e/f8448e06c88e3136a189608597d8bfd7.jpg"

# Add CSS to set the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{image_url}');
        background-size: cover;
        background-position: center;
        height: 100vh;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # To store chat history
if "image_context" not in st.session_state:
    st.session_state.image_context = None  # To store the extracted context from the image
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False  # Track if an image is uploaded

# App UI
st.markdown('<h1 style="color:rgb(0, 0, 0);">Chat with an Image!</h1>', unsafe_allow_html=True)

# Upload file
file_ = st.file_uploader("Upload an Image and Ask Any Question...", type=["jpeg", "jpg", "png"])

if file_:
    # Display uploaded image directly without saving it
    st.image(file_, use_column_width=True)

    # Store the uploaded image and its uploaded status in session state
    st.session_state.image = file_
    st.session_state.image_uploaded = True

    # Extract image context if not already extracted
    if not st.session_state.image_context:
        with st.spinner("Analyzing the image..."):
            st.session_state.image_context = extract_image_context(file_)

# Show text input only if image is uploaded
if st.session_state.image_uploaded:
    # Get user question'
    st.divider()
    user_question = st.text_input("Ask any question about the image:-")

    if user_question:
        with st.spinner("Thinking..."):
            # Combine context and query
            prompt = f"""
            Here is the context extracted from the image:
            {st.session_state.image_context}
            
            User's question: {user_question}
            
            Please answer the question based on the image context and your general knowledge.
            """

            # Generate response
            response = agent.run(prompt)

            # Update chat history
            st.session_state.chat_history.append(("user", user_question))
            st.session_state.chat_history.append(("bot", response))

if file_:
    # Display chat history
    st.divider()
    st.markdown('<h3 style="color:rgb(250, 210, 120);">Chat History...</h3>', unsafe_allow_html=True)

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f"""
                <div style='
                    background-color: lightgray;
                    padding: 10px;
                    border-radius: 10px;
                    margin: 5px 0;
                    text-align: right;
                    color: black;
                '>
                    <b>You:</b> {message}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style='
                    background-color: rgb(80, 80, 80);
                    padding: 10px;
                    border-radius: 10px;
                    margin: 5px 0;
                    text-align: left;
                    color: white;
                '>
                    <b>Bot:</b> {message}
                </div>
                """,
                unsafe_allow_html=True,
            )
