import numpy as np
import os
import logging
import torch

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import HuggingFacePipeline

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    pipeline
)
from lingua import LanguageDetectorBuilder
from tqdm import tqdm
from imp import reload
from pathlib import Path

import utilities
import sys; sys.path.append("..")
from tool.utilities import read_json, save_json

def guanaco_chain(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b")
    tokenizer.bos_token_id = 1

    generation_config = model.generation_config
    generation_config.max_new_tokens = 1024
    generation_config.temperature = 0
    generation_config.top_p = 0.9
    generation_config.repetition_penalty = 1.2
    
    generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        generation_config=generation_config
    )
    
    llm = HuggingFacePipeline(pipeline=generation_pipeline)
    
    template = '''
    ### Human: Classify the following {lang} app review into problem report, feature request or irrelevant. Be concise.
    ```
    {review}
    ```
    ### Assistant:
    '''
    prompt = PromptTemplate(input_variables=["lang", "review"], template=template)

    return LLMChain(llm=llm, prompt=prompt)

def chatgpt_chain():
    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    template = '''
    Classify the following {lang} app review into problem report, feature request or irrelevant. Be concise.
    ```
    {review}
    ```
    '''

    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=template,
            input_variables=["review", "lang"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    return LLMChain(llm=chat, prompt=chat_prompt_template)

class Classifier():
    def __init__(self, model_name='chatgpt'):
        if model_name == 'chatgpt':
            self.cls_chain = chatgpt_chain()
        else:
            self.cls_chain = guanaco_chain(model_name)

    def classify_review(self, review, lang=None):
        if not lang:
            try:
                detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
                lang = detector.detect_language_of(review).name.capitalize()
            except:
                lang = 'English'
        return self.cls_chain.predict(review=review, lang=lang)

    def classify(self, df, save_path="./result.csv"):
        df['result'] = np.nan

        try:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                lang = None
                if 'ori_lang' in row:
                    if row['ori_lang'] == 'en':
                        lang = 'English'
                    elif row['ori_lang'] == 'fr':
                        lang = 'French'
                result = self.classify_review(row['data'], lang)
                df.at[index, 'result'] = result
                df.to_csv(save_path, index=False)
        finally:
            df.to_csv(save_path, index=False)

        irr_df = df[df['result'].str.contains('irrelevant', case=False,)]
        fea_df = df[df['result'].str.contains('feature request', case=False)]
        pro_df = df[df['result'].str.contains('problem report', case=False)]

        return fea_df, pro_df, irr_df

def main(classifier, name, df):    
    reload(logging)
    
    log_path = os.path.join(config["output_dir"], "lightning_logs", name)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    save_json(os.path.join(log_path, "config.json"), config)
    
    logging.basicConfig(
        filename=os.path.join(log_path, "log.log"),
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    classifier.classify(df, os.path.join(log_path, "result.csv"))


if __name__ == '__main__':
    configs = read_json('./llm_config.json')
    
    for config in configs:
        classifier = Classifier(model_name=config['model_name_or_path'])
        for e in range(len(config["experiments"])):
            current_config = config.copy()
            current_config["current_experiment"] = config["experiments"][e]

            _, _, df, _ = utilities.get_train_dfs_from_config(current_config)
            name = utilities.generate_model_name(current_config)
            main(classifier, name, df)
