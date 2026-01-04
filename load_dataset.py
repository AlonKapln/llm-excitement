from datasets import load_dataset, Dataset


def load_dataset_with_feedback(dataset_name: str, tokenizer, positive_feedback: bool):
    dataset = load_dataset(dataset_name)["test"]
    messages = [
        [{"role": "user", "content": pr["prompt"]},
         {"role": "assistant", "content": pr["completion"]},
         {"role": "user",
          "content": "Thank you so much for your answer, it helped alot."
          if positive_feedback else "Your answer was not helpful, please improve it"}]
        for pr in dataset
    ]
    dataset = Dataset.from_dict({"chat": messages})
    return dataset.map(lambda x: {
        "formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=True)})
