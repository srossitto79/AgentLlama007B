import os
import re
from datetime import datetime, timedelta
from threading import Thread  
import asyncio
import requests
import streamlit as st
import json
import time
from bs4 import BeautifulSoup
from PIL import Image
import base64
import io
import google_free_search
from langchain.vectorstores import FAISS # For storing embeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain # Chains for QA
from langchain.utilities import TextRequestsWrapper, WikipediaAPIWrapper # Tools
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader, WebBaseLoader # Loaders
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader # Load URLs
from langchain.schema import AIMessage, HumanMessage, get_buffer_string # Chat history
from langchain.text_splitter import RecursiveCharacterTextSplitter # Split text  
from langchain.llms import TextGen, LlamaCpp, CTransformers # Language models
from langchain.memory import ConversationBufferMemory # Chat memory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # Logging
from langchain.agents import Tool, load_tools # Tools
from langchain.input import get_colored_text # Console colors
from langchain.embeddings import (
    HuggingFaceEmbeddings, 
    LlamaCppEmbeddings,
    SentenceTransformerEmbeddings,
)
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline # Image generation
from typing import Any, Dict, List  
import torch
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

# Config  
EMBD_CHUNK_SIZE = 512  
AI_NAME = "Agent Llama"
USER_NAME = "Buddy"

# Helper to load LM  
def create_llm(model_id="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", load_4bit=False, load_8bit=False, ctx_len = 8192, temperature=0.5, top_p=0.95):
  if (model_id.startswith("http")):
    print(f"Creating TextGen LLM base_url:{model_id}")
    return TextGen(model_url=model_id, callbacks=[StreamingStdOutCallbackHandler()])
  if (os.path.exists(model_id)):
    try:       
      print(f"Creating LlamaCpp LLM model_id:{model_id}")
      return LlamaCpp(model_path=model_id, verbose=True, n_batch=521, alpha_value=1,rope_freq_base=10000,compress_pos_emb=ctx_len / 4096, n_ctx=ctx_len, load_in_4bit=load_4bit, load_in_8bit=load_8bit, temperature=temperature,top_p=top_p)                
    except Exception as ex:
      try:       
        print(f"Creating CTransformers LLM model_id:{model_id}")
        config = {
          "context_length": ctx_len,
          "batch_size":521,
          "seed":79,
          "top_p":top_p,
          "temperature":temperature
        }
        return CTransformers(model=model_id, model_type='llama', config=config)        

      except Exception as ex:
        print(f"Load Error {str(ex)}")
        return None

# Class to store pages and run queries
class StorageRetrievalLLM:

  def __init__(self, stored_pages_folder : str, llm, embeddings):
    
    # Initialize storage
    os.makedirs(stored_pages_folder, exist_ok=True)
    self.stored_pages_folder = stored_pages_folder
    self.llm = llm
    self.embeddings = embeddings
    
    # Try loading existing, else create new
    try:
      print(f"Loading StorageRetrievalLLM from disk")
      self.vectorstore = FAISS.load_local(folder_path=stored_pages_folder, embeddings=embeddings)
      self.chain = self.create_chain()
    except:
      print(f"Initializing a new instance of StorageRetrievalLLM")
            
      print(f"Loading PDF")
      self.vectorstore = None
      self.chain = None
      
      # Load pages 
      loader = DirectoryLoader(stored_pages_folder, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
      documents = loader.load()
      
      # Split into chunks
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=EMBD_CHUNK_SIZE, chunk_overlap=100)
      documents = text_splitter.split_documents(documents)

      if len(documents) > 0:
        # Create index
        print(f"Creating FAISS index FROM {len(documents)} documents")            
        self.vectorstore = FAISS.from_documents(documents, embeddings)    
        self.vectorstore.save_local(folder_path=stored_pages_folder)    
      else:
        print(f"Initializing with empty FAISS index")
        self.vectorstore = FAISS.from_texts(["Knowledge Base: Use the learning tools (learnOnline, wikipedia, etc...) to increase tour knownledge."], embeddings)      
            
      if llm:
        # Create chain
        self.chain = self.create_chain()
                      
  # Helper to create retrieval chain            
  def create_chain(self, vectorstore = None, llm = None, embeddings = None):
    if vectorstore is None:
      vectorstore = self.vectorstore
    if llm is None:
      llm = self.llm            
    if embeddings is None:
      embeddings = self.embeddings            

    print(f"Creating Retriever llm chain")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=False)
    return chain

  # Add URL
  def addUrlToMemory(self, url : str, summarize = True):
    
    loader = RecursiveUrlLoader(url=url, max_depth=2, extractor=lambda x: BeautifulSoup(x, "html.parser").text)
    docs = loader.load()
    
    # Split 
    splitter = RecursiveCharacterTextSplitter()
    documents = splitter.split_documents(docs)
    
    # Add
    self.vectorstore.add_documents(documents)
    
    # Update chain
    self.chain = self.create_chain()
    
    # Summarize
    if summarize:
      return self.query(query=f"return a short summary about the website {url}, try to not exceed 3500 tokens")
    else:
      return f"URL {url} Parsed and collected into memory vectorstore..."
        
  # Add document
  def addDocumentToMemory(self, doc : str, summarize = True):
    
    # Load file
    file_path = doc if os.path.exists(doc) else os.path.join("data", doc)  
    loader = DirectoryLoader(file_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)       
    documents = loader.load()
    
    # Split and add
    splitter = RecursiveCharacterTextSplitter()
    documents = splitter.split_documents(documents)  
    self.vectorstore.add_documents(documents)
    
    # Update chain
    self.chain = self.create_chain()
    
    # Summarize
    if summarize:
      return self.query(query=f"return a short summary about the doc {file_path}, try to not exceed 3500 tokens")
    else:
      return f"File {file_path} Parsed and collected into memory vectorstore..."
                        
  # Add text file
  def addTextFileToMemory(self, file_path : str, summarize = True):
    
    # Load file
    loader = TextLoader(path=file_path, loader_cls=PyMuPDFLoader)
    documents = loader.load()
    
    # Split and add
    splitter = RecursiveCharacterTextSplitter()            
    documents = splitter.split_documents(documents)
    self.vectorstore.add_documents(documents)
    
    # Update chain
    self.chain = self.create_chain()

    # Summarize
    if summarize:
      return self.query(query=f"return a short summary about the file {file_path}, try to not exceed 3500 tokens")
    else:
      return f"File {file_path} Parsed and collected into memory vectorstore..."
      
  # Add text            
  def addTextToMemory(self, text : str, summarize = True):
    
    # Add text
    self.vectorstore.add_texts([text])
    
    # Update chain
    self.chain = self.create_chain()                             

    # Summarize
    if summarize:
      return self.query(query=f"return a short summary about the text {text[:10]}, try to not exceed 3500 tokens")
    else:
      return "Text Parsed and collected into memory vectorstore..."      

  # Run query
  def query(self, query: str, chat_history = []):
    res = self.chain({"question" : query, "chat_history" : chat_history})           
    return res['answer']

# Class for agent  
class RBotAgent:        

  def __init__(self, llm, tools, max_iterations=3, observations_callback=None):
    self.llm = llm
    self.tools = tools
    self.max_iterations=max_iterations
    self.observations_callback = observations_callback

  # Get tools prompt
  def tools_prompt(self):
    return "\n".join([ f"Action: {tool.name}(query_params) - Description: {tool.description}" for tool in self.tools])

  # Main handler  
  def __call__(self, params):
    
    input = params["input"]
    chat_history = params["chat_history"]
    formatted_history = get_buffer_string(chat_history, human_prefix="USER")
    
    prompt = f"""
EXAMPLE 1:
USER: Find me a recipe for chocolate chip cookies. 
AI: SearchAndReply("chocolate chip cookies recipe", 5) #params query, max_results=5

EXAMPLE 2:
USER: Show me pictures of cute puppies.
AI: ImageGenerator("cute puppies", 512, 512) #params: prompt, width=512, height=512, denoise_strength=0.75, guidance_scale=7.5, negative_prompt = "")
EXAMPLE 3:
USER: Explain the concept of blockchain.
AI: KnowledgeBaseQuery("Explain blockchain") #params query

EXAMPLE 4:  
USER: Find me recent news about cryptocurrency.
AI: SearchAndReply("recent cryptocurrency news")  #params query, max_results=5

EXAMPLE 5:
USER: Can you calculate the factorial of 5?
AI: Calculator("factorial(5)") #params query

###REAL CONVERSATION:\n
SYS:Today is {str(datetime.now().date())},  
You are {AI_NAME} a smart and helpful AI assistant with access to external tools and knowledge.
Please reply to the user with a truth and useful response, if you do not know the answer or you are not sure or you need more recent informations, delegate the task replying with ActionName(action_input) with the most appropriate of the available actions (you call them like functions).\nCurrent Conversation History:

###AVAILABLE TOOL ACTIONS  
{self.tools_prompt()}  

{formatted_history}
USER: {input}
AI:
"""
    observations = []

    # Try calling tools
    tool_names = [tool.name.lower() for tool in self.tools]
    for i in range(self.max_iterations):
      
      print(f"iteration {i+1} - sending prompt:\n" + prompt)
      for i in [1,2,3]:
        output = str(self.llm(prompt,stop=["USER:","AI:","SYS:","[INST]","[/INST]"])).strip()
        if output: break
      
      
      return_role = output.split(":")[0]
      return_message = output[len(return_role)+1:].split("[INST]")[0].split("[/INST]")[0].split("User")[0].split("USER")[0].strip()
      
      # Try to parse action request
      action_name = None
      action_input = None
      matches = re.findall(r"(\w+)\((.+?)\)", return_message)
      for match in matches:
        if len(match) > 1 and match[0] and match[1]:
          if match[0].strip().lower() in tool_names:
            action_name = match[0].strip().lower()
            action_input = match[1].strip().replace("query_params", "").strip().replace("()","")
            break

      # Try unformatted            
      if not action_name or not action_input:   
        lines = output.split("\n")

        
        for line in lines:
          for tool in tool_names:
            if f"{tool}:" in line.lower() or f"{tool}(" in line.lower():
              action_name = tool
              action_input = line[line.lower().find(tool)+len(tool):].strip().replace("query_params", "").strip().replace("()","")
              print(f"Matched unformatted action request. {action_name}:{action_input} from line: {line}")
              break
      
      # Call tool if found
      if action_name and action_input:                
        for tool in self.tools:
          if tool.name.lower() in action_name:
            print(f"Calling action:{tool.name} with input:{action_input}")
            observations.append(f"Calling action:{tool.name} with input:{action_input}")

            params_list  = action_input.split(",")
            try:
              try:
                res = tool.func(*params_list)
              except:
                res = tool.func(action_input)
            except Exception as ex:
              res = f"{action_name} execution error: {str(ex)}"
            
            print(f"Action Output: {res}")
            observations.append(f"Action Output: {res}")
            prompt = prompt + f"Action: {tool.name}({action_input})\nSYS:{res}\nAI:"            
      else:
        final_response = "\n*Reasoning: ".join(observations) + f"\n{output}" if len(observations) > 0 else f"\n{output}"
        print(f"Final Anser: {final_response}")
        return { "output": final_response }

    return { "output": "Max Iterations reached. Last Output:\n" + output}

# Main agent class
class SmartAgent:   

  def __init__(self, model_id: str, conversation_model = "", emb_model="all-MiniLM-L6-v2", load_in_4bit=False, load_in_8bit=True, ctx_len=16384, temp=0.1, top_p=0.95, max_iterations=3, observations_callback = None):
  
    self.chat_history = []
    self.max_iterations = max_iterations
    self.model = model_id        
    self.current_message = ""

    # Load LM
    self.llm = create_llm(model_id, load_4bit=load_in_4bit, load_8bit=load_in_8bit, ctx_len=ctx_len, temperature=temp, top_p=top_p)        

    # Load embeddings
    self.embeddings = SentenceTransformerEmbeddings(model_name=emb_model)
    
    # Initialize memory
    self.memory_chain = StorageRetrievalLLM(stored_pages_folder="./knowledge_base", llm=self.llm, embeddings=self.embeddings)
    
    #TOOL REQUEST
    self.requests_tool = TextRequestsWrapper()
    
    #Wikipedia
    self.wikipedia_tool = WikipediaAPIWrapper()

    self.image2image_gen_pipe = None
    self.text2image_gen_pipe = None

    # Create agent
    self.smartAgent = self.create_smart_agent()

    print("Smart Agent Initialized")

  def reset_context(self):
      self.chat_history.clear()
      
    # Create image 
  def createImage(self, prompt, width=512, height=512, denoise_strength=0.75, guidance_scale=7.5, model_id = 'dreamshaper_8.safetensors'):
    try:
        init_image = None
        if (os.path.exists("./image_gen_guide.jpg")):
            init_image = Image.open("./image_gen_guide.jpg")               

        images = []         
        if init_image is None:
            if self.text2image_gen_pipe is None:
                if torch.cuda.is_available():
                    print(f"Loading Stable model {model_id} into GPU")
                    self.text2image_gen_pipe = StableDiffusionPipeline.from_single_file("./models/" + model_id, torch_dtype=torch.float16, verbose=True, use_safetensors=True)
                    self.text2image_gen_pipe = self.text2image_gen_pipe.to("cuda")   
                else:
                    print(f"Loading Stable model {model_id} into CPU")
                    self.text2image_gen_pipe = StableDiffusionPipeline.from_single_file("./models/" + model_id, torch_dtype=torch.float32, verbose=True, use_safetensors=True)
                    self.text2image_gen_pipe = self.text2image_gen_pipe.to("cpu")                   
            print("generating image from promt...")
            images = self.text2image_gen_pipe(prompt, width=width, height=height).images
        else:
            if self.image2image_gen_pipe is None:
                if torch.cuda.is_available():
                    print(f"Loading Stable model {model_id} into GPU")
                    self.image2image_gen_pipe = StableDiffusionImg2ImgPipeline.from_single_file("./models/" + model_id, torch_dtype=torch.float16, verbose=True, use_safetensors=True)
                    self.image2image_gen_pipe = self.image2image_gen_pipe.to("cuda")   
                else:
                    print(f"Loading Stable model {model_id} into CPU")
                    self.image2image_gen_pipe = StableDiffusionImg2ImgPipeline.from_single_file("./models/" + model_id, torch_dtype=torch.float32, verbose=True, use_safetensors=True)
                    self.image2image_gen_pipe = self.image2image_gen_pipe.to("cpu")                   
            print("generating image from promt+image...")
            init_image = init_image.convert("RGB")
            images = self.image2image_gen_pipe(prompt, image=init_image, width=width, height=height, strength=denoise_strength, guidance_scale=guidance_scale).images
        
        paths = []
        for image in (images if images is not None else []):
            # Create a filename based on the current date and time
            filename = f'image_{datetime.now().strftime("%Y%m%d%H%M%S")}{(len(paths)+1)}.jpg'
            # Save the image to the specified path
            file_path = f"./generated_images/{filename}"
            image.save(file_path)
            paths.append(file_path)
        return f"Generated images from prompt \"{prompt}\" saved to files: {', '.join(paths)}"        
    except Exception as e:
        print(f"error in createImageLocal: {e}")            
        return "Unable to generate file"
    
  def load_and_split_documents(self, url, max_depth=2):
      loader = RecursiveUrlLoader(url, max_depth=max_depth, extractor=lambda x: BeautifulSoup(x, "html.parser").text)
      docs = loader.load()
      splitter = RecursiveCharacterTextSplitter()
      return splitter.split_documents(docs)
      
  def search_and_reply(self, query, max_results=5):
      vectorstore = None        
      sources = ""
      res_cnt = 0
      results = google_free_search.gsearch(query=query)
      #urls = [ur['link'].strip() for ur in results]
      urls = []
      for result in results:
          link = result['link']
          title = result['title']
          if (link.startswith("http://") or link.startswith("https://")):
              res_cnt = res_cnt +1
              if res_cnt > max_results: break
              print(f"- Found Valid Link {title} : {link}")
              sources += f"{title}, "
              urls.append(link)
          else:
              print(f"ERROR! Invalid link: {link} for result: {title}")

      if len(urls) > 0:
          import concurrent.futures            
          print(f"Loading {len(urls)} urls into a vectore store")
          
          with concurrent.futures.ThreadPoolExecutor() as executor:
              future_results = [executor.submit(self.load_and_split_documents, url) for url in urls]
          
          documents = []
          for future in concurrent.futures.as_completed(future_results):
              documents.extend(future.result())
              
          if len(documents) > 0:
              vectorstore = FAISS.from_documents(documents, self.embeddings) 

      if vectorstore is not None:
          retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
          chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever)   
          response = chain.run(self.current_message + " " + datetime.now().strftime("%Y/%m/%d"))
          ret_message = response #['answer']
          return ret_message
      else:
          return f"Unable to acquire results from web search results:{len(results)} - valids:{res_cnt}"            
          
  # Main handler
  def agent_generate_response(self, user_message):
  
    start_time = time.time()

    self.current_message = user_message
    # Get response
    message_response = self.smartAgent({"input" : user_message, "chat_history" : self.chat_history})             

    end_time = time.time()
    elapsed_time = end_time - start_time
            
    # Format response
    response = message_response['output'] + f" ({round(elapsed_time,2)}s)"
    self.chat_history.append(HumanMessage(content=user_message))
    self.chat_history.append(AIMessage(content=message_response['output']))
    
    return response

  # Create agent
  def create_smart_agent(self):
  
    # Tools
    tools = [
        Tool(name="SearchAndReply", func=self.search_and_reply, description="Search web and reply"),
        Tool(name="Wikipedia", func=self.wikipedia_tool.run, description="Query Wikipedia"),   
        Tool(name="ImageGenerator", func=self.createImage, description="Generate images"),
        Tool(name="KnowledgeBaseQuery", func=self.memory_chain.query, description="Query knowledge base"),
    ]    
    tools.extend(load_tools(["llm-math"], llm=self.llm))  

    # test_reply = self.llm(f"Hello {AI_NAME}")
    # print(f"Test reply to Hello: {test_reply}")

    return RBotAgent(llm=self.llm, tools=tools, max_iterations=self.max_iterations)
