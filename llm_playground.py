import streamlit as st
from ollama_client import EtanOllamaClient
from doc_process import process_document
from chroma_db_manager import ChromaDBManager
from langchain.prompts import ChatPromptTemplate
from llm_templates import TemplateManager
from settings_manager import SettingsManager
import os
import torch

# this bizarre torch hack is needed to get the torch classes module to work properly
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

class CapturePromptCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.raw_prompts = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n\n ### RAW PROMPT ####")
        print(prompts)
        # 'prompts' is a list of the raw prompt strings (or messages) sent to the LLM.
        self.raw_prompts.extend(prompts)

already_ran_once = False

st.title("LLM Playground")
st.subheader("Upload a document and ask questions about it.")

# Initialize components

chromadb_manager = ChromaDBManager()
template_manager = TemplateManager()
settings_manager = SettingsManager()
ollama_client = EtanOllamaClient(api_url=settings_manager.get_setting('ollama_api_url'))

# ---- SETTINGS SECTION ----
st.sidebar.header("Settings")

# Load settings
saved_settings = settings_manager.settings

# Embeddings model selection
embeddings_models = ["multi-qa-mpnet-base-cos-v1", "all-MiniLM-L6-v2"]
default_embeddings_index = embeddings_models.index(saved_settings['embeddings_model']) if saved_settings['embeddings_model'] in embeddings_models else 0
embeddings_model = st.sidebar.selectbox(
    "Select an embeddings model", 
    embeddings_models,
    index=default_embeddings_index
)

# Save setting whenever changed
if embeddings_model != saved_settings['embeddings_model']:
    settings_manager.update_setting('embeddings_model', embeddings_model)

chromadb_manager = ChromaDBManager(embeddings_model_name=embeddings_model)

# LLM selection
llm_options = ollama_client.list_models()
default_llm_index = llm_options.index(saved_settings['selected_llm']) if saved_settings['selected_llm'] in llm_options else 0
selected_llm = st.sidebar.selectbox(
    "Select an LLM", 
    llm_options,
    index=default_llm_index
)

# Save setting whenever changed
if selected_llm != saved_settings['selected_llm']:
    settings_manager.update_setting('selected_llm', selected_llm)

# Add a horizontal separator
st.sidebar.markdown("---")

# ---- PROMPT TEMPLATES SECTION ----
st.sidebar.header("Prompt Templates")

# Initialize template state if not present in session state
if 'is_new_template' not in st.session_state:
    st.session_state.is_new_template = False
if 'new_template_name' not in st.session_state:
    st.session_state.new_template_name = ""
if 'new_template_description' not in st.session_state:
    st.session_state.new_template_description = ""  
if 'new_system_template' not in st.session_state:
    st.session_state.new_system_template = ""
if 'new_human_template' not in st.session_state:
    st.session_state.new_human_template = ""

# Load available templates or create default if none exist
available_templates = template_manager.list_templates()
if not available_templates:
    default_template = template_manager.get_first_template()
    available_templates = [default_template]

template_names = [t['name'] for t in available_templates]
template_files = [t['file'] for t in available_templates]

# Get the saved template or default to first one
saved_template_name = saved_settings['selected_template']
default_template_index = next(
    (i for i, t in enumerate(available_templates) if t['name'] == saved_template_name), 
    0
)

# Only show the template selection if not creating a new template
if not st.session_state.is_new_template and template_names:
    # Add template selection session state if not present
    if 'selected_template_index' not in st.session_state:
        st.session_state.selected_template_index = default_template_index
        
    # Use a key for the selectbox and define a callback to update session state
    def on_template_change():
        # Update selected template only when the dropdown is changed
        selected_idx = st.session_state.template_dropdown
        st.session_state.selected_template_index = selected_idx
        
        # Save setting whenever template changes
        selected_template = available_templates[selected_idx]
        template_name = selected_template['name']
        if template_name != saved_settings['selected_template']:
            settings_manager.update_setting('selected_template', template_name)
    
    # Show selectbox with current selected template from session state
    selected_template_index = st.sidebar.selectbox(
        "Select Template",
        range(len(template_names)),
        format_func=lambda i: template_names[i],
        index=st.session_state.selected_template_index,
        key="template_dropdown",
        on_change=on_template_change
    )
    
    # Get the selected template from the session state index to ensure consistency
    selected_template = available_templates[st.session_state.selected_template_index]
    
    # Extract values from the selected template
    system_template = selected_template['system_template']
    human_template = selected_template['human_template']
    template_name = selected_template['name']
    template_description = selected_template['description']
    selected_template_file = selected_template['file']

# Function to handle new template creation
def create_new_template():
    st.session_state.is_new_template = True
    st.session_state.new_template_name = "New Template"
    st.session_state.new_template_description = ""
    st.session_state.new_system_template = "You are a helpful AI assistant answering questions based on the context.\n\nContext: {context}"
    st.session_state.new_human_template = "Question: {question}"

# Function to cancel new template creation
def cancel_new_template():
    st.session_state.is_new_template = False

# New Template button on its own line
if not st.session_state.is_new_template:
    if st.sidebar.button("New Template"):
        create_new_template()
else:
    # Display cancel button when in new template mode
    if st.sidebar.button("Cancel New Template"):
        cancel_new_template()

# Set template variables based on mode
if st.session_state.is_new_template:
    template_name = st.session_state.new_template_name
    template_description = st.session_state.new_template_description
    system_template = st.session_state.new_system_template
    human_template = st.session_state.new_human_template
else:
    # These were already set from the selected template above
    pass

# Template editing
template_name = st.sidebar.text_input("Template Name", value=template_name, key="template_name_input")
template_description = st.sidebar.text_input("Description", value=template_description, key="template_desc_input")

system_template = st.sidebar.text_area(
    "System Template", 
    value=system_template,
    height=150,
    key="system_template_input"
)

human_template = st.sidebar.text_area(
    "Human Message Template", 
    value=human_template,
    height=75,
    key="human_template_input"
)

# Update session state with current values if in new template mode
if st.session_state.is_new_template:
    st.session_state.new_template_name = template_name
    st.session_state.new_template_description = template_description
    st.session_state.new_system_template = system_template
    st.session_state.new_human_template = human_template

# Template variables info
st.sidebar.markdown("**Available Variables:**")
st.sidebar.markdown("- `{context}`: Retrieved document content")
st.sidebar.markdown("- `{question}`: User's question")

# Save template button
save_col1, save_col2 = st.sidebar.columns([1, 1])
with save_col1:
    if st.button("Save Template"):
        template_data = {
            'name': template_name,
            'description': template_description,
            'system_template': system_template,
            'human_template': human_template
        }
        
        # Create a filename from the template name
        filename = template_name.lower().replace(' ', '_') + '.yaml'
        if template_manager.save_template(template_data, filename):
            st.sidebar.success(f"Template saved as {filename}")
            # Update the selected template in settings
            settings_manager.update_setting('selected_template', template_name)
            # Reset new template mode if we were in it
            if st.session_state.is_new_template:
                st.session_state.is_new_template = False
            # Force refresh of available templates
            st.rerun()

with save_col2:
    if not st.session_state.is_new_template and len(template_names) > 1 and st.button("Delete"):
        if template_manager.delete_template(selected_template_file):
            st.sidebar.success(f"Template {template_name} deleted")
            # Force refresh
            st.rerun()

# Option to add custom variables
custom_vars = {}
use_custom_vars = st.sidebar.checkbox("Add Custom Variables")

if use_custom_vars:
    custom_var_name = st.sidebar.text_input("Variable Name (without {})")
    custom_var_value = st.sidebar.text_input("Variable Value")
    if st.sidebar.button("Add Variable") and custom_var_name:
        if 'custom_vars' not in st.session_state:
            st.session_state.custom_vars = {}
        st.session_state.custom_vars[custom_var_name] = custom_var_value

if 'custom_vars' in st.session_state:
    st.sidebar.markdown("**Custom Variables:**")
    for var, value in st.session_state.custom_vars.items():
        st.sidebar.markdown(f"- `{{{var}}}`: {value}")
        custom_vars[var] = value
        
        # Allow removing variables
        if st.sidebar.button(f"Remove {var}"):
            del st.session_state.custom_vars[var]
            st.rerun()

# Store template config for use in the chat logic
template_config = {
    "system_template": system_template,
    "human_template": human_template,
    "custom_vars": custom_vars
}

# Add a horizontal separator
st.sidebar.markdown("---")

# ---- COLLECTIONS SECTION ----
st.sidebar.header("Available Collections")

# File upload handling - moved here from above
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "txt"])
if uploaded_file is not None:
    file_path = f"./uploads/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    processed_document = process_document(file_path, 200, 20)
    chromadb_manager.add_document(document_id=uploaded_file.name, text=processed_document)

# Get all available collections from ChromaDB
available_collections = chromadb_manager.list_collections()

# Map of collection names to actual collection objects
collection_map = {str(coll): coll for coll in available_collections}
collection_names = list(collection_map.keys())

# Get the saved collections from settings (these are stored as strings)
saved_collection_names = saved_settings['selected_collections']
if not isinstance(saved_collection_names, list):
    saved_collection_names = []

# Display collections with checkboxes
selected_collection_names = []
for collection_name in collection_names:
    # Check if this collection name was previously selected
    default_checked = collection_name in saved_collection_names
    if st.sidebar.checkbox(collection_name, value=default_checked):
        selected_collection_names.append(collection_name)

# Convert selected collection names back to actual collection objects for querying
selected_collections = [collection_map[name] for name in selected_collection_names if name in collection_map]

# Save selected collection names to settings
if set(selected_collection_names) != set(saved_collection_names):
    settings_manager.update_setting('selected_collections', selected_collection_names)

# Number of articles slider
default_num_articles = saved_settings['num_articles']
num_articles = st.sidebar.slider(
    "Number of articles", 
    min_value=1, 
    max_value=20, 
    value=default_num_articles
)

# Save setting whenever changed
if num_articles != saved_settings['num_articles']:
    settings_manager.update_setting('num_articles', num_articles)

with st.container(border=True):
    db_query = st.text_input("Database Query")
    if st.button("Query Database"):
        db_response = chromadb_manager.query_database(db_query, from_collections=selected_collections, n_results=num_articles)
        st.write(db_response)

with st.container(border=True):
    answer_field = st.text("")
    if 'log' in st.session_state:
        answer_field = st.text(st.session_state.log)

    if st.button("Start new chat"):
        answer_field.write("")
        if 'log' in st.session_state:
            del st.session_state['log']
        if 'messages' in st.session_state:
            del st.session_state['messages']
        if 'custom_vars' in st.session_state:
            del st.session_state['custom_vars']

    # Query field
    user_query = st.text_input("Ask a question")
    submit_button = st.button("Submit")
    if submit_button:
        # Instantiate your callback handler and manager
        capture_handler = CapturePromptCallbackHandler()
        callback_manager = CallbackManager([capture_handler])
        
        # https://python.langchain.com/docs/integrations/chat/ollama/
        # https://python.langchain.com/docs/tutorials/rag/#overview
        llm = ollama_client.get_langchain_ollama_model(selected_llm, callback_manager=callback_manager)
        docs = []
        messages = []
        # Add a new key to session state if it doesn't exist
        # Session State also supports attribute based syntax
        if 'messages' not in st.session_state:
            docs = chromadb_manager.query_database(user_query, from_collections=selected_collections, n_results=num_articles)
            docs_str = "\n".join(docs)
            
            # Using the prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", template_config["system_template"]),
                ("human", template_config["human_template"])
            ])
            
            # Format the prompt with the actual values
            template_vars = {
                "context": docs_str,
                "question": user_query
            }
            
            # Add any custom variables
            if template_config["custom_vars"]:
                template_vars.update(template_config["custom_vars"])
            
            messages = prompt.format_messages(**template_vars)
            
        else:
            messages = st.session_state.messages
            
            # For follow-up questions, we can also use a template
            human_prompt = ChatPromptTemplate.from_messages([
                ("human", template_config["human_template"])
            ])
            
            # Format with the question
            template_vars = {"question": user_query}
            if template_config["custom_vars"]:
                template_vars.update(template_config["custom_vars"])
                
            follow_up_message = human_prompt.format_messages(**template_vars)[0]
            messages.append(follow_up_message.model_dump())

        answer = llm.invoke(messages)
        
        # If using format_messages(), we get Message objects, but session_state needs dicts
        if not isinstance(messages[0], dict):
            # Convert any Message objects to dictionaries for storage
            messages = [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in messages]
            
        messages.append({"role": "assistant", "content": f"Answer: {answer.content}"})
        st.session_state.messages = messages

        if 'log' not in st.session_state:
            st.session_state.log = ""
        
        st.session_state.log += f"{user_query}\n-----------------\n{answer.content}\n\n"
        answer_field.write(st.session_state.log)
        