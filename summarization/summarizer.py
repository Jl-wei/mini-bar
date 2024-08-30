import os
import openai
import time
import numpy as np
import pandas as pd
import socket
import sys; sys.path.append("..")
import torch
import nltk
from pathlib import Path
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    OpenAIGPTTokenizer,
    BitsAndBytesConfig, 
    StoppingCriteria, 
    StoppingCriteriaList
)
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from lingua import Language, LanguageDetectorBuilder

nltk.download('punkt')

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check the id of line break in llama tokenizer
        # self.tokenizer.convert_tokens_to_ids(["<0x0A>"])
        stop_token_ids = [0, 13]
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class SummarizerModel:
    def __init__(self, model_name):
        self.model_name = model_name
        openai.api_key = os.environ['OPENAI_API_KEY']
        if model_name == 'chatgpt':
            self.tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
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
            self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b")
            self.tokenizer.bos_token_id = 1
            self.stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    def generate(self, prompt):
        if self.model_name == 'chatgpt':
            condition = True
            while condition:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[
                                {"role": "system", "content": "You are a helpful assistant that summarizes app reviews."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                    condition = False
                except (openai.error.ServiceUnavailableError, 
                        openai.error.RateLimitError, 
                        ConnectionError, 
                        socket.timeout) as e:
                    print(e)
                    time.sleep(30)

            return response['choices'][0]['message']['content']
        else:
            formatted_prompt = "You are a helpful assistant that summarizes app reviews."
            formatted_prompt += f"### Human: {prompt} ### Assistant:" 
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")
            outputs = self.model.generate(inputs=inputs.input_ids, 
                                    #  stopping_criteria=self.stopping_criteria,
                                     max_new_tokens=1024, 
                                     temperature=0,
                                     top_p=0.9,
                                     repetition_penalty=1.2,
                                     )
            answer_ids = outputs[0][len(inputs["input_ids"][0]):-1]

            return self.tokenizer.decode(answer_ids, skip_special_tokens=True)
    
    def token_length(self, text):
        return len(self.tokenizer.tokenize(text))

class ExtractiveSummarizer:
    def __init__(self):
        self.embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        languages = [Language.ENGLISH, Language.FRENCH]
        self.lang_detector = LanguageDetectorBuilder.from_languages(*languages).build()

    def summarize(self, reviews):
        sents = self.convert_to_leng_sents(reviews)
        embeddings = self.embedder.encode(sents, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings, embeddings)
        similarity_scores = list(map(lambda cs: sum(cs), cosine_scores.tolist()))
        
        similarity_scores_sorted_indice = np.argsort(similarity_scores)[::-1]
        for indice in similarity_scores_sorted_indice:
            if self.is_english(sents[indice]):
                return sents[indice]

        # If there is no English sentence, 
        # use the French sentence with the largest similarity score
        return sents[similarity_scores_sorted_indice[0]]
    
    def is_english(self, text):
        return self.lang_detector.detect_language_of(text) == Language.ENGLISH

    def convert_to_leng_sents(self, reviews):
        sents = []
        for rew in reviews:
            r_sents = sent_tokenize(rew)
            for sent in r_sents:
                if len(sent.split()) >= 5:
                    sents.append(sent)
        return sents

class AbstractiveSummarizer:
    def __init__(self, model_name='chatgpt'):
        self.model = SummarizerModel(model_name=model_name)
        self.max_length = 2000

    def summarize(self, reviews):
        reviews = list(reviews)
        if self.token_count(reviews) <= self.max_length:
            summary = self.summarize_reviews_list(reviews)
            # summary = ".".join(summary.split(".")[0:-1])
            return summary
        else:
            sub_summaries = []
            for chunk in self.split_into_chunks(reviews):
                sub_summaries.append(self.summarize_reviews_list(chunk))
            return self.summarize(sub_summaries)
    
    # Split a list of reviews into small list with a maximum token count self.max_length
    def split_into_chunks(self, reviews):
        review_chunks = []
        
        chunk_length = 0
        chunk = []
        for r in reviews:
            if chunk_length + self.token_count(r) > self.max_length:
                review_chunks.append(chunk)
                chunk_length = self.token_count(r)
                chunk = [r]
            else:
                chunk_length += self.token_count(r)
                chunk.append(r)
        return review_chunks
    
    def token_count(self, text):
        total = 0
        if isinstance(text, str):
            total = self.model.token_length(text)
        elif isinstance(text, list):
            for t in text:
                total += self.model.token_length(t)
        return total

    def summarize_reviews_list(self, reviews):
        prompt = "Please summarize all following app reviews into one English sentence\n"
        prompt += "```\n"
        prompt += "\n".join(map(lambda r: '- {}'.format(r), list(reviews)))
        prompt += "\n```"
        
        return self.model.generate(prompt)


if __name__ == "__main__":
    summarizer = AbstractiveSummarizer(model_name='chatgpt')
    # summarizer = AbstractiveSummarizer(model_name="timdettmers/guanaco-33b-merged")
    # summarizer = ExtractiveSummarizer()
    
    data_dir = "../dataset/for_clustering/labelled"
    output_dir = "./output"
    column = 'ground_truth'

    # This script will generate the summaries for the clustering dataset
    path = Path(__file__).parent.joinpath(data_dir).resolve()
    for app_folder in path.iterdir():
        print('=======================', app_folder.name, '=======================')
        output_path = Path(__file__).parent.joinpath(output_dir, app_folder.name).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        for file in app_folder.iterdir():
            if not str(file).endswith('.csv'): continue
            df = pd.read_csv(file)
            output_df = pd.DataFrame(columns=['document', 'summary'])
            for group in df[column].dropna().unique():            
                reviews = df[df[column] == group].data.tolist()

                if len(reviews) < 5: continue

                document = " ||||| ".join(reviews)
                summary = summarizer.summarize(reviews)
                line = pd.DataFrame([[document, summary]], columns=['document', 'summary'])
                output_df = pd.concat([output_df, line])
                print('***********', group, '***********')

            output_df.to_csv(Path.joinpath(output_path, file.name), index=False)
