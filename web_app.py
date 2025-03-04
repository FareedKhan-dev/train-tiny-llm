"""
Step-by-Step Guide to Building This LLM:
Medium Article: https://medium.com/@fareedkhandev/building-a-perfect-million-parameter-llm-from-scratch-in-python-3b16e26b4139
"""

import random
import re

import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="30M-SFT-LLM", initial_sidebar_state="collapsed")

# Custom CSS to style buttons and layout
st.markdown("""
    <style>
        .stButton button {
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;
            margin: 5px 10px 5px 0 !important;
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Model Configuration
system_prompt = []
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to process assistant responses
def format_assistant_response(content):
    content = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', content)  # Adjusts line spacing
    return content

@st.cache_resource
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    return model.eval().to(device), tokenizer

def clear_chat():
    st.session_state.messages = []
    st.session_state.chat_messages = []

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Sidebar for Model Configuration
st.sidebar.title("Model Settings")
st.sidebar.text("Higher context memory may reduce response quality over long conversations.")
st.session_state.history_chat_num = st.sidebar.slider("Number of Historical Dialogues", 0, 6, 0, step=2)
st.session_state.max_new_tokens = st.sidebar.slider("Max Sequence Length", 256, 8192, 256, step=1)
st.session_state.top_p = st.sidebar.slider("Top-P", 0.8, 0.99, 0.85, step=0.01)
st.session_state.temperature = st.sidebar.slider("Temperature", 0.6, 1.2, 0.85, step=0.01)

MODEL_PATHS = {
    # beautiful_model_name: "./your_model_path",
    "30M-SFT-LLM": "./30M-SFT-LLM"
}
selected_model = st.sidebar.selectbox('Select Model', list(MODEL_PATHS.keys()), index=0)
model_path = MODEL_PATHS[selected_model]

avatar_url = "https://avatars.githubusercontent.com/u/63067900"
slogan = f"Hi, I'm {selected_model}"

st.markdown(f"""
    <div style="text-align: center;">
        <img src="{avatar_url}" style="width: 45px; height: 45px;">
        <h2>{slogan}</h2>
        <p style="color: #bbb;">You can create your own 30 Million Parameter LLM using <a href="https://medium.com/@fareedkhandev/building-a-perfect-million-parameter-llm-from-scratch-in-python-3b16e26b4139">my Medium article</a>.</p>
    </div>
""", unsafe_allow_html=True)

def main():
    model, tokenizer = load_model_tokenizer(model_path)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=avatar_url):
                st.markdown(format_assistant_response(message["content"]), unsafe_allow_html=True)
                if st.button("🗑", key=f"delete_{i}"):
                    st.session_state.messages = st.session_state.messages[:i-1]
                    st.session_state.chat_messages = st.session_state.chat_messages[:i-1]
                    st.rerun()
        else:
            st.markdown(f'<div style="text-align: right;"><div style="display: inline-block; background-color: gray; color: white; padding: 8px 12px; border-radius: 10px;">{message["content"]}</div></div>', unsafe_allow_html=True)
    
    user_input = st.chat_input(placeholder="Send a message to 30M-SFT-LLM")
    
    if user_input:
        st.markdown(f'<div style="text-align: right;"><div style="display: inline-block; background-color: gray; color: white; padding: 8px 12px; border-radius: 10px;">{user_input}</div></div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant", avatar=avatar_url):
            placeholder = st.empty()
            setup_seed(random.randint(0, 2 ** 32 - 1))
            
            conversation_history = system_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
            formatted_prompt = tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)[-(st.session_state.max_new_tokens - 1):]
            
            input_tensor = torch.tensor(tokenizer(formatted_prompt)['input_ids'], device=device).unsqueeze(0)
            with torch.no_grad():
                generated_responses = model.generate(input_tensor, tokenizer.eos_token_id, max_new_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature, top_p=st.session_state.top_p, stream=True)
                
                full_response = ""
                for response in generated_responses:
                    decoded_text = tokenizer.decode(response[0].tolist(), skip_special_tokens=True)
                    if not decoded_text or decoded_text[-1] == '�':
                        continue
                    full_response = decoded_text.replace(formatted_prompt, "")
                    placeholder.markdown(format_assistant_response(full_response), unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
