
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig



if __name__ == "__main__":
    test_json = "my_test_1_90.json"
    label_xlsx = "data/shunfeng/label-v3.xlsx"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = "/home/ma-user/work/model/Baichuan2-7B-Chat/Baichuan2-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

    model.generation_config = GenerationConfig.from_pretrained(model_dir)
    
    TP = TN = FP = FN = 0

    with open(test_json, "r", encoding='utf-8') as f:
        data = json.load(f)
        for idx, item in tqdm(enumerate(data), desc="Processing json", total=len(data)):
            prompt = item["instruction"] + item["input"]
            messages = [{"role": "user", "content": prompt}]
            response = model.chat(tokenizer, messages)
            gt_res = item["output"]
            
            # 初始化计数器

            # 计算 TP, TN, FP, FN
            for true_label, pred_label in zip(gt_res, response):
                if true_label == "包含" and pred_label == "包含":
                    TP += 1
                elif true_label == "不包含" and pred_label == "不包含":
                    TN += 1
                elif true_label == "不包含" and pred_label == "包含":
                    FP += 1
                elif true_label == "包含" and pred_label == "不包含":
                    FN += 1

            if idx+1 % 100 == 0:
                # 计算准确率
                accuracy = (TP + TN) / (TP + TN + FP + FN)

                # 计算召回率
                recall = TP / (TP + FN)
                print(f'Samples: {idx+1} items, Accuracy: {accuracy:.2f}, Recall: {recall:.2f}')
        
        # 计算准确率
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        # 计算召回率
        recall = TP / (TP + FN)
        print(f'Final Results: Accuracy: {accuracy:.2f}, Recall: {recall:.2f}')
