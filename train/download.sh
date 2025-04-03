mkdir -p ../train_data
cd ../train_data

huggingface-cli download ubowang/ScholarCopilot-TrainingData --local-dir . --repo-type dataset

