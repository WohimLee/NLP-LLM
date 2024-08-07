

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto
model_name_or_path = "/home/ma-user/work/01440561/01449344/output/checkpoint-168"
model = AutoModelForCausalLM.from_pretrained(
  model_name_or_path,
  torch_dtype="auto",
  device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

while True:

    prompt = input("请输入:")
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    generated_ids = model.generate(
      model_inputs.input_ids,
      max_new_tokens=400,
      repetition_penalty=1.15
    )
    generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)