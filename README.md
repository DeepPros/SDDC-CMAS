# SDDC-CMAS
Synthetic Data-Driven Classification of Chinese Medical Abstract Sentence
* DocSCAN-main folder
1. run finetune.py to get the finetuned model
2. use the fine-tuned model to run DocSCAN following DocSCAN-main/README.pd
Note: see the source code from "https://github.com/dominiksinsaarland/DocSCAN"
* LiDA-main foder
1. choose train and test datasets in sbert/main.py
2. improved MEC logic is in sbert/sbert.py
Note: see the source code from "https://github.com/yest/LiDA"
* Bert-CTC-main folder
1. create dataset packages for three datasets as the format of example 'THUCNews' 
2. in run.py, choose which dataset to run
3. run algorithms following README.pd
4. Note: see the source code from "https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch" 
* CTC-main folder
1. create dataset packages for three datasets as the format of example 'THUCNews' 
2. in run.py, choose which dataset to run
3. run algorithms following README.pd 
Note: see the source code from "https://github.com/649453932/Chinese-Text-Classification-Pytorch"
