from datasets import load_dataset
import sglang as sgl
import asyncio
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run inference with a specific model path.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs.jsonl",
    )
    args = parser.parse_args()

    dataset = load_dataset("tatsu-lab/alpaca_eval", data_files="alpaca_eval.json", split="eval", trust_remote_code=True)
    model_path = args.model_path

    # TODO: initialize sglang egnine here
    # you may want to explore different args we can pass here to make the inference faster
    # e.g. dp_size, mem_fraction_static
    
    
    # TESTED ON 2 x H100 and forced mem_fraction_static to 0.2 to make sure it works on low memory of 16GB

    llm = sgl.Engine(
          model_path=model_path,
          tp_size=1,
          dp_size=2,
          mem_fraction_static=0.2,
          attention_backend="dual_chunk_flash_attn",
          context_length=16384,
          trust_remote_code=True,
          disable_cuda_graph=True
      )

    prompts = []

    for i in dataset:
        prompts.append(i['instruction'])

    sampling_params = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8192}

    outputs = []

    # TODO: you may want to explore different batch_size
    ### FOR 2x H100
    batch_size = 16

    from tqdm import tqdm
    for i in tqdm(range(0, len(prompts), batch_size)):
        # TODO: prepare the batched prompts and use llm.generate
        # save the output in outputs
        batch_prompts = prompts[i:i + batch_size]

          # Generate outputs for the batch
        batch_outputs = llm.generate(
            batch_prompts,
            sampling_params=sampling_params
        )

        # Extract text from outputs
        for output in batch_outputs:
            outputs.append(output["text"])


    with open(args.output_file, "w") as f:
        for i in range(0, len(outputs), 10):
            instruction = prompts[i]
            output = outputs[i]
            f.write(json.dumps({
                "output": output,
                "instruction": instruction
            }) + "\n")

if __name__ == "__main__":
    main()
