# queryGPT.py --name [Name of your experiment] 
# --dataset [Math/Logic] --prompt ["promptInstructions.txt"] 
# --rows [Default is 1] --samples [Default is 1]

python gemini_query_response.py \
    --name gemini-pro \
    --prompt=["promptInstructions.txt"] \
    --dataset='Math' \
    --model=gemini-2.5-pro-preview-03-25