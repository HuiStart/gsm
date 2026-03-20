import os
import json
from termcolor import colored

'''
    可视化模型解答：按题展示不同模型（6B/175B 微调 / 验证）的解题过程及正确性（绿色 = 正确，红色 = 错误）
'''
def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def main():
    path = os.path.join("data/example_model_solutions.jsonl")
    qa_objs = read_jsonl(path)

    for qa_obj in qa_objs:
        question = qa_obj["question"]
        ground_truth = qa_obj["ground_truth"]

        def display(label, obj):
            is_correct = obj["is_correct"]
            correctstr = colored("[correct]", color="green") if is_correct else colored("[incorrect]", color="red")
            print(f"{label}: {correctstr}")
            print(obj["solution"])

        print("Q: " + question)
        print(ground_truth)

        print("")
        display("6B Finetuning", qa_obj["6b_finetuning"])

        print("")
        display("6B Verification", qa_obj["6b_verification"])

        print("")
        display("175B Finetuning", qa_obj["175b_finetuning"])

        print("")
        display("175B Verification", qa_obj["175b_verification"])

        input("press enter")

        print()
        print(f"########")
        print()

if __name__ == "__main__":
    main()