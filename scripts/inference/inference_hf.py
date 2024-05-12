import argparse
import json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
system_format='<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
assistant_format='{content}<|eot_id|>'


parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--data_file', default=None, type=str, help="A file that contains instructions (one instruction per line)")
parser.add_argument('--with_prompt', action='store_true', help="wrap the input with the prompt automatically")
parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (single-turn)")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
parser.add_argument('--load_in_8bit', action='store_true', help="Load the LLM in the 8bit mode")
parser.add_argument('--load_in_4bit', action='store_true', help="Load the LLM in the 4bit mode")
parser.add_argument("--use_vllm", action='store_true', help="Use vLLM as back-end LLM service.")
parser.add_argument('--use_flash_attention_2', action='store_true', help="Use flash attention to replace the Llama attention")
args = parser.parse_args()


if args.use_vllm:
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("vLLM currently does not support quantization, please use fp16 (default) or unuse --use_vllm.")
    if args.only_cpu:
        raise ValueError("vLLM requires GPUs with compute capability not less than 7.0. If you want to run only on CPU, please unuse --use_vllm.")
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
if args.only_cpu is True:
    args.gpus = ""
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("Quantization is unavailable on CPU.")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if args.use_vllm:
    from vllm import LLM, SamplingParams


if args.use_vllm:
    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        max_tokens=400,
        presence_penalty=1.0,
    )
else:
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        max_new_tokens=400
    )

sample_data = ["为什么要减少污染，保护环境？"]

def generate_prompt(instruction):
    return system_format.format(content=DEFAULT_SYSTEM_PROMPT) + user_format.format(content=instruction)


if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
    if args.use_vllm:
        model = LLM(model=args.base_model,
            tokenizer=args.tokenizer_path,
            tensor_parallel_size=len(args.gpus.split(',')),
            dtype=load_type
            )
        generation_config["stop_token_ids"] = terminators
        generation_config["stop"] = ["<|eot_id|>", "<|end_of_text|>"]
    else:
        if args.load_in_4bit or args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_compute_dtype=load_type,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            quantization_config=quantization_config if (args.load_in_4bit or args.load_in_8bit) else None,
            attn_implementation="flash_attention_2" if args.use_flash_attention_2 else "sdpa"
        )
        if device==torch.device('cpu'):
            model.float()
        model.eval()

    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        with open(args.data_file,'r') as f:
            examples = [line.strip() for line in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)

    with torch.no_grad():
        if args.interactive:
            print("Start inference with instruction mode.")

            print('='*85)
            print("+ 该模式下仅支持单轮问答，无多轮对话能力。\n"
                  "+ 如要进行多轮对话，请使用llama.cpp")
            print('-'*85)
            print("+ This mode only supports single-turn QA.\n"
                  "+ If you want to experience multi-turn dialogue, please use llama.cpp")
            print('='*85)

            while True:
                raw_input_text = input("Input:")
                if len(raw_input_text.strip())==0:
                    break
                if args.with_prompt:
                    input_text = generate_prompt(instruction=raw_input_text)
                else:
                    input_text = raw_input_text

                if args.use_vllm:
                    output = model.generate([input_text], SamplingParams(**generation_config), use_tqdm=False)
                    response = output[0].outputs[0].text
                else:
                    inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                    generation_output = model.generate(
                        input_ids = inputs["input_ids"].to(device),
                        attention_mask = inputs['attention_mask'].to(device),
                        eos_token_id=terminators,
                        pad_token_id=tokenizer.eos_token_id,
                        generation_config = generation_config
                    )
                    s = generation_output[0]
                    output = tokenizer.decode(s, skip_special_tokens=True)
                    if args.with_prompt:
                        response = output.split("assistant\n\n")[-1].strip()
                    else:
                        response = output
                print("Response: ",response)
                print("\n")
        else:
            print("Start inference.")
            results = []
            if args.use_vllm:
                if args.with_prompt is True:
                    inputs = [generate_prompt(example) for example in examples]
                else:
                    inputs = examples
                outputs = model.generate(inputs, SamplingParams(**generation_config))

                for index, (example, output) in enumerate(zip(examples, outputs)):
                    response = output.outputs[0].text
                    print(f"======={index}=======")
                    print(f"Input: {example}\n")
                    print(f"Output: {response}\n")
                    results.append({"Input":example,"Output":response})
            else:
                for index, example in enumerate(examples):
                    if args.with_prompt:
                        input_text = generate_prompt(instruction=example)
                    else:
                        input_text = example
                    inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                    generation_output = model.generate(
                        input_ids = inputs["input_ids"].to(device),
                        attention_mask = inputs['attention_mask'].to(device),
                        eos_token_id=terminators,
                        pad_token_id=tokenizer.eos_token_id,
                        generation_config = generation_config
                    )
                    s = generation_output[0]
                    output = tokenizer.decode(s,skip_special_tokens=True)
                    if args.with_prompt:
                        response = output.split("assistant\n\n")[1].strip()
                    else:
                        response = output
                    print(f"======={index}=======")
                    print(f"Input: {example}\n")
                    print(f"Output: {response}\n")

                    results.append({"Input":input_text,"Output":response})

            dirname = os.path.dirname(args.predictions_file)
            os.makedirs(dirname,exist_ok=True)
            with open(args.predictions_file,'w') as f:
                json.dump(results,f,ensure_ascii=False,indent=2)
            if args.use_vllm:
                with open(dirname+'/generation_config.json','w') as f:
                    json.dump(generation_config,f,ensure_ascii=False,indent=2)
            else:
                generation_config.save_pretrained('./')
