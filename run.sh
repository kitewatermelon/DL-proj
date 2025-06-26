python run.py --version cleaned --cuda 0 --opt 1 2>&1 | tee logs_cleaned.txt
python baseline_model.py 
python ttest.py