# This code is modified from C-Eval Project: https://github.com/SJTU-LIT/ceval

import os
import re
from tqdm import tqdm
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
system_format='<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
assistant_format='{content}<|eot_id|>'


class Llama_Evaluator():
    def __init__(self, choices, k, model_path, device, use_flash_attention_2=False, verbose=False):
        load_type = torch.float16
        self.choices = choices
        self.k = k
        self.device = device
        self.verbose = verbose
        self.use_flash_attention_2 = use_flash_attention_2
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            attn_implementation="flash_attention_2" if self.use_flash_attention_2 else "sdpa"
            )
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_k=0,
            top_p=1.0,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.1,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True
        )

        self.sA_id = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.sB_id = self.tokenizer.encode("B", add_special_tokens=False)[0]
        self.sC_id = self.tokenizer.encode("C", add_special_tokens=False)[0]
        self.sD_id = self.tokenizer.encode("D", add_special_tokens=False)[0]
        self.A_id = self.tokenizer.encode("：A")[-1]
        self.B_id = self.tokenizer.encode("：B")[-1]
        self.C_id = self.tokenizer.encode("：C")[-1]
        self.D_id = self.tokenizer.encode("：D")[-1]


    def eval_subject(self, subject_name,
            test_df,
            dev_df=None,
            few_shot=False,
            save_result_dir=None,
            with_prompt=False,
            do_test=False):
        all_answers = {}

        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        history = f"以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n\n"
        if few_shot:
            if with_prompt:
                history = self.generate_alpaca3_few_shot_prompt(history, dev_df, subject=subject_name)
            else:
                history = self.generate_llama3_few_shot_prompt(history, dev_df)
        answers = ['NA'] * len(test_df) if do_test is True else list(test_df['Answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, few_shot=False)
            instruction = history + question
            if with_prompt:
                if few_shot:
                    instruction = history + user_format.format_map({'content': question})
                else:
                    instruction = system_format.format(content=DEFAULT_SYSTEM_PROMPT) + user_format.format_map({'content': instruction})
            inputs = self.tokenizer(instruction, return_tensors="pt")
            terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            generation_output = self.model.generate(
                    input_ids = inputs["input_ids"].to(self.device),
                    attention_mask = inputs['attention_mask'].to(self.device),
                    eos_token_id=terminators,
                    pad_token_id=self.tokenizer.eos_token_id,
                    generation_config = self.generation_config
                )

            logits = generation_output.scores[0][0]
            logits = logits.float().cpu().detach()
            choices1_logits = logits[[self.sA_id,self.sB_id,self.sC_id,self.sD_id]]
            choices2_logits = logits[[self.A_id,self.B_id,self.C_id,self.D_id]]
            choicesAll_logits = (choices1_logits + choices2_logits).numpy()
            assert not (np.any(np.isinf(choicesAll_logits)) or np.any(np.isnan(choicesAll_logits)))
            ans = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choicesAll_logits)]
            response = self.tokenizer.decode([logits.argmax(-1).item()])
            if ans == answers[row_index]:
                correct_num += 1
                correct = 1
            else:
                correct = 0
            if self.verbose is True:
                print(f"\n======={str(row_index)}=======")
                print(f"question: {question}\n")
                print(f"instruction: {instruction}\n")
                print(f"response: {response}\n")
                print(f"extracted answer: {ans}")
                print(f"ground truth: {answers[row_index]} \n")
            if save_result_dir:
                result.append(response)
                score.append(correct)

            all_answers[str(row_index)] = ans

        correct_ratio = 100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output'] = result
            test_df['correctness'] = score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_test.csv'))

        return correct_ratio, all_answers

    def format_example(self, line, few_shot=False):
        example = line['Question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        if few_shot:
            example += '\n答案：' + line["Answer"] + '\n\n'
        else:
            example += '\n答案：'
        return example

    def generate_llama3_few_shot_prompt(self, history, dev_df):
        prompt = history
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(dev_df.iloc[i, :], few_shot=True)
        return prompt

    def generate_alpaca3_few_shot_prompt(self, history, dev_df, subject=None):
        prompt = history
        prompt_template = (
            "<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "好的，我会结合{subject}相关知识回答<|eot_id|>"
        )

        prompt = prompt_template.format_map({'instruction':prompt, 'system_prompt':DEFAULT_SYSTEM_PROMPT, 'subject':subject})
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            line = dev_df.iloc[i, :]
            q=line['Question']
            for choice in self.choices:
                q += f'\n{choice}. {line[f"{choice}"]}'

            a = line['Answer']
            q += "\n答案："
            prompt += user_format.format(content=q) + assistant_format.format(content=a)
        return prompt
