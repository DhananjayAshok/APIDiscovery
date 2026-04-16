To run:

1. git clone --recursive <this_repo>

2. Populate the required configs/private_vars.yaml placeholders

```bash
bash runs/create.sh
bash runs/setup.sh
```

Run the RL experiment:
```
bash rl.sh --model_name meta-llama/Llama-3.1-8B-Instruct
```