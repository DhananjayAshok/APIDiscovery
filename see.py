import pandas as pd

method="zeroshot"
dataset="humaneval"
judge="Llama-3.1-8B-Instruct"
model="Meta-Llama-3-8B-Instruct"

path = f"results/{dataset}/{method}_{model}-{dataset}-judge-{judge}.jsonl"

df = pd.read_json(path, lines=True)

def l(row):
    if 'description' in row:
        description = row['description']
    else:
        description = row['true_description']
    #train_inputs = row['train_inputs']
    predicted_description = row['predicted_description']
    score_output = row['score_output']
    score = row['score']
    n_queries = row['n_queries']
    concluded = row['concluded']
    #print(f"Train Inputs: {train_inputs}")
    print(f"Description: {description}")
    print(f"Number of Queries: {n_queries}")
    print(f"Concluded: {concluded}")    
    print(f"Predicted Description: {predicted_description}")
    print(f"Score: {score}")
    return

def d(n=1):
    for i in range(n):
        row = df.sample(1).iloc[0]
        l(row)
        print("\n\n XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n\n")

if __name__ == "__main__":
    d()



