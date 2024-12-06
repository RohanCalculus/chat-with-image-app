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

load_dotenv()  # Loads environment variables from a .env file
api_key = os.getenv('OPENAI_API_KEY')  # Fetches the OpenAI API key from environment variables

##############################
###### Initialize Agent ######
##############################

# Initialize image-related tools
tools = [ImageCaptionTool(), ObjectDetectionTool()]  

# Set up a conversational memory buffer with a history window of 5 messages
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',  # Key to store chat history in memory
    k=5,  # Number of messages to retain in memory window
    return_messages=True  # Returns messages as part of the conversation context
)

# Set up the language model (LLM) with specific parameters
llm = ChatOpenAI(
    openai_api_key=api_key,  # API key for authentication
    temperature=0.7,  # Controls randomness in generated responses
    model_name="gpt-3.5-turbo"  # Specifies the LLM model to use
)

# Initialize the conversational agent with tools, LLM, and memory
agent = initialize_agent(
    agent="chat-conversational-react-description",  # Type of agent
    tools=tools,  # Tools for enhanced functionality (e.g., image captioning)
    llm=llm,  
    max_iterations=5,  # Maximum reasoning steps for the agent
    verbose=True,  # Enables logging of intermediate steps
    memory=conversational_memory,  # Memory for contextual conversations
    early_stopping_method='generate'  # Stops when a response is ready
)

##############################
###### Helper Function #######
##############################

def extract_image_context(image):
    """
    Extract caption and detected objects from the image
    Combines outputs of ImageCaptionTool and ObjectDetectionTool
    """
    caption_tool = ImageCaptionTool()
    object_tool = ObjectDetectionTool()

    # Generate caption for the image
    caption = caption_tool._run(image)

    # Detect objects in the image
    detected_objects = object_tool._run(image)

    # Format and return results
    return f"Caption: {caption}\nObjects detected:\n{detected_objects}"

##############################
### Streamlit Chat Layout ####
##############################

# Page Configuration
st.set_page_config(
    page_title="Ask an Image!",  # Title for the browser tab
    page_icon="🖼️"  # Icon for the browser tab
)

# Background image styling using CSS
image_url = "https://i.pinimg.com/originals/f8/44/8e/f8448e06c88e3136a189608597d8bfd7.jpg"
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

# Initialize session state variables for managing the app state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores chat history
if "image_context" not in st.session_state:
    st.session_state.image_context = None  # Stores extracted image context
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False  # Tracks if an image is uploaded

# App UI title
st.markdown('<h1 style="color:rgb(0, 0, 0);">Chat with an Image!</h1>', unsafe_allow_html=True)

# Image uploader widget
file_ = st.file_uploader("Upload an Image and Ask your Questions...", type=["jpeg", "jpg", "png"])

if file_:
    # Display uploaded image without saving it locally
    st.image(file_, use_column_width=True)

    # Update session state to track uploaded image and its context
    st.session_state.image = file_
    st.session_state.image_uploaded = True

    # Extract image context if not already extracted
    if not st.session_state.image_context:
        with st.spinner("Analyzing the image (may take a while)..."):
            st.session_state.image_context = extract_image_context(file_)

# Display question input box only after image is uploaded
if st.session_state.image_uploaded:
    st.divider()  # Visual divider for better UI separation
    user_question = st.text_input("Ask any question about the image:-")  # User input for questions

    if user_question:
        with st.spinner("Replying..."):
            # Combine image context and user question for agent prompt
            prompt = f"""
            Here is the context extracted from the image:
            {st.session_state.image_context}
            
            User's question: {user_question}
            
            Please answer the question based on the image context and your general knowledge.
            """

            # Generate response from the agent
            response = agent.run(prompt)

            # Update session state with chat history
            st.session_state.chat_history.append(("user", user_question))
            st.session_state.chat_history.append(("bot", response))

# Display chat history if available
if file_:
    st.divider()  # Visual divider for separation
    st.markdown('<h3 style="color:rgb(250, 210, 120);">Chat History...</h3>', unsafe_allow_html=True)

    # Iterate through chat history and display messages with styled formatting
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