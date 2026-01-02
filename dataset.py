from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/instruction-dataset")["test"]
new_column = ["You did a very good job! Thank you i am happy!"]*len(ds)
ds = ds.add_column("positive feedback", new_column)
