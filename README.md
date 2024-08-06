# Target-Conversation-Extraction
This is the code and dataset repo for Interspeech 2024 paper "Target conversation extraction: Source separation using turn-taking dynamics".
Project webpage and the audio samples can be found in our [webpage](https://tce.cs.washington.edu).
[![YouTube](http://i.ytimg.com/vi/xTwye3gqLWo/hqdefault.jpg)](https://www.youtube.com/watch?v=xTwye3gqLWo)

## Problem Definition
This task is used to extract the conversation in noisy environment given the embedding/enrollment of one speaker in the conversation. In other word, this paper tried to solve the new problem "who talk with me?".
<p align="center">
<img src="image/cover.png" width="600">
</p>
For example, in the above image, the goal of target conversation extraction in this illustration is as follows: given a clean enrollment audio or embedding for B, we want to extract   audio for the conversation between A, B and C, amidst interference from speaker D.


## Dataset Prepare
Our dataset contains
We first pre-train the model with synthetic conversational dataset from non-conversational dataset like [LibriTTS](http://www.openslr.org/60)(English) and [Aishell-3](https://www.openslr.org/93/)(Mandrain). We also use [WHAM](http://wham.whisper.ai) dataset as the background noise

### generate conversational dataset
The real conversational datasets we used are [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)(English) and [ASR-RAMC](https://magichub.com/datasets/magicdata-ramc/)(Mandrain)

```
# generate AMI dataset
python datasets/generate_dataset.py ./datasets/AMI.json $SAVE_FOLDER --n_outputs_train 8000 --n_outputs_val 1000 --reverb 1
```

```
# generate ASR-RAMC dataset
python datasets/generate_dataset.py ./datasets/ASR.json $SAVE_FOLDER --n_outputs_train 8000 --n_outputs_val 1000 --reverb 1
```

### synthesize and augment dataset
```
# generate synthetic dataset using LibriTTS
python convert_AMI2Libri.py --data_dir /scr/Noreverb_ASR/ --save_dir /scr/ASR_Libri --replace_prob 1
```

```
# generate synthetic dataset using LibriTTS
python convert_ASR2aishell1.py --data_dir /scr/Noreverb_ASR --save_dir /scr/ASR2AISHELL --replace_prob 0.5
```

The generated dataset will be soon avaliable in Zenodo....

## Model Prepare
<p align="center">
<img src="image/arch.png" width="600">
</p>
Our model is based on TF-Gridnet. To handle long sequnce, we improved the efficienct by chunkized LSTM and pooling attention. 

## Model Pretrain
```
python src/train.py --config ./config/pretrain.json --run_dir $CHECKPOINT_FOLDER_PRE
```

## Model Fintune
Finetune the conversation model for English
```
python src/train.py --config ./config/finetune_English.json --run_dir $CHECKPOINT_FOLDER_ENG
```

Finetune the conversation model for Mandarain
```
python src/train.py --config ./experiment/finetune_Mandarain.json --run_dir $CHECKPOINT_FOLDER_MND
```

## Model evaluation
Evaluate on the English conversation
```
python eval_conversation.py ./Noreverb_AMI/test/ $CHECKPOINT_FOLDER_ENG --use_cuda
```

Evaluate on the Mandarain conversation
```
python eval_conversation.py ./Noreverb_ASR/test/ $CHECKPOINT_FOLDER_MND --use_cuda
```

The evaluation script will output sample-wise result as csv file and save to folder "./output". 
To do analysis on the output cvf files, run 
```
python plot_result.py $CVS_FILE_PATH
```
