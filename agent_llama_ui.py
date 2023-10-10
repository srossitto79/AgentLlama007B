import streamlit as st
from streamlit_chat import message
import os
import io
from dotenv import load_dotenv
import requests
import glob
import json
import shutil
from RBotReloaded import SmartAgent
import time
from PIL import Image
from langchain.schema import AIMessage, HumanMessage

load_dotenv()


default_model = ""
default_context = 8192
default_load_type = "Auto"
default_iterations = 2
default_temperature = 0.5
default_topp = 0.95

@st.cache_resource
def agent(model, temperature, top_p, context_length, load_8bit, load_4bit, max_iterations):    
    ag = SmartAgent(f"./models/{model}" if os.path.exists(f"./models/{model}") else model, temp=temperature, top_p=top_p, load_in_4bit=load_4bit, load_in_8bit=load_8bit, ctx_len=context_length, max_iterations=max_iterations) if model else None
    st.session_state["temperature_executive"] = temperature
    st.session_state["max_iterations_executive"] = max_iterations
    st.session_state["model_executive"] = model
    st.session_state["context_length_executive"] = context_length
    st.session_state["load_options_executive"] = "Load 4-bit" if load_8bit else "Load 4-bit" if load_4bit else "Auto"
    st.session_state["top_p_executive"] = top_p

    return ag
    
def get_models():
    supported_extensions = ["bin","pth","gguf"]
    models_directory = "./models"  # Replace with the actual path
    # Use os.listdir to get a list of filenames in the directory
    models = os.listdir(models_directory)
    # Filter out any subdirectories, if any
    models = [model for model in models if (model.lower().split(".")[-1] in supported_extensions) and os.path.isfile(os.path.join(models_directory, model))]  
    models.append("http://localhost:5000")    
    return models

def current_agent():
    model = st.session_state.get("model", default_model)
    temperature = st.session_state.get("temperature", default_temperature)
    max_iterations = st.session_state.get("max_iterations", default_iterations)
    context_length = st.session_state.get("context_length", default_context)
    load_options = st.session_state.get("load_options", default_load_type)    
    top_p = st.session_state.get("top_p", default_topp)
    
    model = st.session_state.get("model_executive", model)
    temperature = st.session_state.get("temperature_executive", temperature)
    max_iterations = st.session_state.get("max_iterations_executive", max_iterations)
    context_length = st.session_state.get("context_length_executive", context_length)
    load_options = st.session_state.get("load_options_executive", load_options)    
    top_p = st.session_state.get("top_p_executive", top_p)
    
    return agent(model, temperature, top_p, context_length, load_options=="Load 8-bit", load_options=="Load 4-bit", max_iterations)

def history():
    return [] if current_agent() is None else current_agent().chat_history

#@st.cache_data
def generate_text(input):
    start_time = time.time()
    output = "Error: Model not Loaded!" if current_agent() is None else current_agent().agent_generate_response(input)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n----------------------")
    print(f"Agent Reply: {output} - Input: {input}")
    print(f"Elapsed Time: {elapsed_time} seconds")
    print(f"Agent Reply: {output}")
    print(f"\n----------------------")
    return output + f" ({round(elapsed_time,2)}s)"


def get_generated_files():
    # Specify the directory path where the generated images are stored
    directory = "./generated_images"

    # Get the list of files in the directory
    files = glob.glob(f"{directory}/*.jpg")  # Modify the file extension as per your generated image format

    # Return the list of file paths
    return files

# Function to list files in the "./knowledge_base/" folder
def list_files_in_knowledge_base_folder():
    knowledge_base_folder = "./knowledge_base/"
    files = os.listdir(knowledge_base_folder)
    return [file for file in files if os.path.isfile(os.path.join(knowledge_base_folder, file))]

# Function to add a file to the "./knowledge_base/" folder
def add_file_to_knowledge_base(file):
    knowledge_base_folder = "./knowledge_base/"
    final_path = os.path.join(knowledge_base_folder, file.name)
    
    with open(final_path, "wb") as f:
        f.write(file.read())

    if current_agent() is None:
        st.error("Model Not Loaded!")
    else:
        current_agent().memory_chain.addDocumentToMemory(os.path.join(knowledge_base_folder, file.name))

# Function to add a file to the "./knowledge_base/" folder
def set_image_gen_guide(file): 
    bytes_data = io.BytesIO(file.read())
    image = Image.open(bytes_data)
    image = image.convert("RGB")
    image.save("./image_gen_guide.jpg")
    
def unset_image_gen_guide():    
    if os.path.exists("./image_gen_guide.jpg"):
        os.remove("./image_gen_guide.jpg")
        
def get_index_size():        
    index_file_path = "./knowledge_base/index.faiss"  # Replace with the actual path to your index file
    if os.path.exists(index_file_path):
        index_size = os.path.getsize(index_file_path)
        return index_size / 1024
    else:
        print(f"{index_file_path} does not exist or is not accessible.")
        return 0
        
# @cl.langchain_factory(use_async=True)
# def factory():
#     return current_agent().smartAgent

def render_simple_chat():
    models = get_models()
    models.append("")

    model = st.session_state.get("model", default_model)
    temperature = st.session_state.get("temperature", default_temperature)
    max_iterations = st.session_state.get("max_iterations", default_iterations)
    context_length = st.session_state.get("context_length", default_context)
    load_options = st.session_state.get("load_options", default_load_type)    
    top_p = st.session_state.get("top_p", default_topp)

    with st.sidebar:
        st.image("./avatar.png")
        st.sidebar.title("LLM Options")
        max_iterations = st.sidebar.slider("Max Iterations", min_value=1, max_value=10, step=1, key="max_iterations")
        model = st.selectbox(label="Model", options=models, key="model")
        if (not model.startswith("http")):
            temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, step=0.1, key="temperature")
            top_p = st.sidebar.slider("top_p", min_value=0.1, max_value=1.0, step=0.1, key="top_p")
            context_length = st.sidebar.slider("Context Length", min_value=1024, max_value=131072, step=1024,  key="context_length")
            # Load Options
            load_options = st.sidebar.radio("Load Options", ["Auto", "Load 4-bit", "Load 8-bit"], key="load_options")

        if (st.sidebar.button("Apply Changes to Model")):
            st.session_state["temperature_executive"] = temperature
            st.session_state["max_iterations_executive"] = max_iterations
            st.session_state["model_executive"] = model
            st.session_state["context_length_executive"] = context_length
            st.session_state["load_options_executive"] = load_options
            st.session_state["top_p_executive"] = top_p
            #st.experimental_rerun()

        if st.sidebar.button("Reset Chat Context", disabled=not (current_agent() is not None and len(current_agent().chat_history) > 0)) and current_agent() is not None:
            current_agent().reset_context()

    st.sidebar.write("-----")

    st.sidebar.title("Documents Context")
    st.sidebar.subheader(f"Current Memory Size {round(get_index_size() / 1024,2)}MB")

    uploaded_file = st.sidebar.file_uploader("Drag and Drop a File to ./knowledge_base/", type=["txt", "pdf", "docx"])     
    
    if st.sidebar.button("Reset Long Term Memory", disabled=not (current_agent() is not None and get_index_size() > 0)) and current_agent() is not None:
        current_agent().reset_knowledge()

    st.sidebar.write("-----")

    st.sidebar.title("Images Generation")

    if os.path.exists("./image_gen_guide.jpg"):
        st.sidebar.image("./image_gen_guide.jpg")
        if st.sidebar.button("Remove Image Generation Guidance"):
            unset_image_gen_guide()
            st.experimental_rerun()
    else:
        image_gen_guide = st.sidebar.file_uploader("Drag and Drop an image for the image generation", type=["jpg", "png"])  
        if image_gen_guide:
            set_image_gen_guide(image_gen_guide)
            st.sidebar.success(f"File '{image_gen_guide.name}' set as image generation guidance.")
    
    if uploaded_file:
        add_file_to_knowledge_base(uploaded_file)
        st.sidebar.success(f"File '{uploaded_file.name}' added to Knowledge Base.")
        
    with st.sidebar:
        #GENERATED FILES
        generated_files = get_generated_files()
        st.sidebar.subheader("Generated Files")
        for file_path in generated_files:      
            st.write("---")
            st.write(file_path.split("/")[-1].split("\\")[-1])      
            st.image(file_path)

    i = 0
    for m in history():
        i = i +1
        gen = str(m.content)
        #saved to files: ./generated_images/image_202310091819331.jpg
        if str(gen).endswith(".jpg") and os.path.exists(gen.split(" ")[-1]):
            st.image(gen.split(" ")[-1])

        message(gen, is_user=m.type.lower() == "human", key=str(i))
        
    user_input = st.chat_input("Prompt", key="input_text")
    if user_input:
        message(user_input, is_user=True, key=str(i+1))
        res = generate_text(user_input)
        message(res, is_user=False, key=str(i+2))         
            

##### BEGIN MAIN #####
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
            
if 'model' not in st.session_state:
    st.session_state['model'] = default_model
    st.session_state['temperature'] = default_temperature
    st.session_state['max_iterations'] = default_iterations
    st.session_state['context_length'] = default_context
    st.session_state['load_options'] = default_load_type
    st.session_state['top_p'] =  default_topp
    
st.set_page_config(page_title="Agent Llama", page_icon="🤖", layout="wide")

st.title("Agent Llama")

render_simple_chat()