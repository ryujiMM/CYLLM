import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-medium", device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-medium")

st.title('CYLLM')

# メインコンテンツのコンテナ
main_container = st.container()

# 入力プロンプト
with main_container:
    prompt = st.text_input("プロンプトを入力してください", "")
    token = st.text_input("長さを数字で入力してください。50~200くらいでどうぞ", 50)
    token = int(token)
if prompt:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=token,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    st.write(output)
