#!/usr/bin/env bash

declare -A ARGS
ARGS["p"]="left"   # padding default
ARGS["t"]="400"     # max_new_tokens default
ARGS["b"]="8"     # batch_size default
ARGS["n"]="" # OpenAI default batch name
ARGS["g"]="" # generation_complete and output_perplexity suffices
ARGS["r"]="" # ignore checkpoint argument. I know its badly named. 

# Required arguments
REQUIRED_ARGS=("i" "o" "m" "c" "d")

# Help function
usage() {
    echo "Usage: $0 -m model_name -i input_file -o output_file -c input_column -d output_column [-p padding] [-b batch_size] [-t max_new_tokens] [-n openai_batch_name] [-g generation_suffix] [-r ignore_checkpoint]"
    echo "Required:"
    echo "  -m model_name     Name of the model to use"
    echo "  -i input_file      Path to the input file" 
    echo "  -o output_file     Path to the output file"   
    echo "  -c input_column    Name of the input column in the file"
    echo "  -d output_column    Name of the output column in the file"    
    echo "Options:"
    echo "  -p padding          Padding type"
    echo "  -b batch_size       Batch size"
    echo "  -t max_new_tokens   Maximum new tokens to generate"
    echo "  -n openai_batch_name OpenAI batch name"
    echo "  -g generation_suffix Suffix for generation_complete column"
    echo "  -r ignore_checkpoint  If set, ignores checkpointing and runs inference in one go"
    exit 1
}

# Parse flags
while getopts ":m:i:c:d:b:t:p:n:o:g:r:" opt; do
    case $opt in
        m|i|c|b|t|p|n|o|d|g|r)
            ARGS["$opt"]="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Check required arguments
MISSING=false
for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then
        echo "Error: Missing required argument -$req"
        MISSING=true
    fi
done

if [[ "$MISSING" == true ]]; then
    usage
fi

# Print active variables
echo "Active variables in infer.sh:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

if [[ -z "${ARGS["r"]}" ]]; then
    ignore_checkpoint_arg=""
else
    ignore_checkpoint_arg="--ignore_checkpoint"
fi

if [[ -n "${ARGS["n"]}" ]]; then
    echo "Running OpenAI inference"
    bash scripts/llm_utils.sh python infer.py --model_name ${ARGS["m"]} --input_file ${ARGS["i"]} --output_file ${ARGS["o"]} --input_column ${ARGS["c"]} --output_column ${ARGS["d"]} --generation_complete_column "inference_complete"${ARGS["g"]} --checkpoint_every 0.1 --max_new_tokens ${ARGS["t"]} $ignore_checkpoint_arg openai --batch_name ${ARGS["n"]}
else
    echo "Running local inference"
    bash scripts/llm_utils.sh python infer.py --model_name ${ARGS["m"]} --input_file ${ARGS["i"]} --output_file ${ARGS["o"]} --input_column ${ARGS["c"]} --output_column ${ARGS["d"]} --generation_complete_column "inference_complete"${ARGS["g"]} --checkpoint_every 0.1 --max_new_tokens ${ARGS["t"]} $ignore_checkpoint_arg hf --batch_size ${ARGS["b"]} --padding_side ${ARGS["p"]} 
fi


