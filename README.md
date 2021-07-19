Part of 1st place solution for Coleridge Initiative - Show US the Data.

# Summary
This solution is a text extraction model with CLM backbone, in this case GPT, and beam search. The reason a unidirectional language model like GPT worked better than models like BERT is that bidirectional models can find a shortcut by looking to the future, they don't need to care about the context and will try to find a substring that most resemble a dataset name, thus lead to overfitting.

In my GPT + beamsearch model, GPT will be forced to predict **whether the next token would be the start of dataset name**, given only the previous context, and then predict **wherether the next token would be the end of the mention**, given the starting point and the extracted content so far. To make it more robust, I used a few sources of dataset and replaced 95% of mentions in the original training set with these random labels.

# Modeling 
## Model architecture
For each input sequence we want to predict the start tokens and their respective end tokens of dataset mentions.
In training, each training sample only has one start position and one end position.
We first predict the start position using Softmax + CrossEntropy on the features of each input token.

![Predicting start token](https://i.imgur.com/Ln4HvmJ.png)

For training, we ignore the predicted probabilities and take the feature from the ground truth token, for inference we take the one with highest probability. We then concatenate that embedding to all the other extracted embeddings and use that new features to predict the position of the end token i.e. predicting the end position with regard to the start token.
![enter image description here](https://i.imgur.com/BUInoqT.png)

## Beam search
The architecture can be modified to predict multiple mentions:
 -   We take top-k hidden states corresponding to top-k start indices with highest probabilities, this is normalized with sigmoid, not softmax like training.
 -   Each hidden state is then concatenated into the representations at every position.
 -   The new representation is fed to a MLP, similar to training. We then select top-k end indices for each selected hidden state, resulting in k*k top start-end pairs.
 -   We then calculate the joint probabilities of every start-end pairs and take any pairs with a score large than 0.8.
# Training
## Preparing training data
Please refer to the `preprocess` notebook. Running the whole notebook should generate all the data needed for training.
## Training
Run this command for training:
```
$ python train.py --model gpt2  --train_path ../input/pickled/train_aug_gpt_256.pkl --lr 2e-5 \
--batch_size 8 --accumulation_steps 16 --epochs 7 --seed 13
```
Parameters:
 - `model`: The transformer architecture, this should work with all variants of `gpt` that `huggingface` supports.
 - `train_path`: Path to the pickled training data generated with the `preprocessing` notebook
 - `batch_size` and `accumulation_steps`: For `gpt2` I used 8 and 16, for `gpt2-medium` they are 2 and 48 respectively.
 - `epochs`: Training epochs
 - `lr`: Learning rate
 - `seed`: Random seed for reproducibility, this seems to be bugged.

# Inference

Run this command for inference.
```
$ python infer.py --ckpt_path ./models/gpt2.bin --input_path ./test_article.txt \ 
--batch_size 24 --beam_size 10 --threshold 0.8 --max_sequence_length 128 --overlap 32
```
Parameters:

 - `ckpt_path`: Path to the trained checkpoint.
 - `input_path`: Path to the input article, should be in plaintext form.
 - `batch_size`: Inference batch size
 - `beam_size`: Beam search width
 - `threshold`: Threshold for the joint probability of a start-end pair to be considered valid
 - `max_sequence_length`: The size of the sliding window when splitting the input data, doesn't have to be the same as training. Shorter for better recall and longer for better precision.
 - `overlap`: Number of overlapping tokens between consecutive windows.

The output should looks like this:
![Output example](https://i.imgur.com/71wDkNF.png)
