mkdir -p ../data
cd ../data

huggingface-cli download TIGER-Lab/ScholarCopilot-Data-v1 --local-dir . --repo-type dataset

mkdir -p ../model_v1208
cd ../model_v1208

huggingface-cli download TIGER-Lab/ScholarCopilot-v1 --local-dir . --repo-type model

