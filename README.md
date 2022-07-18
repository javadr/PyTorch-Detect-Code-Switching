# PyTorch-Detect-Code-Switching

## Task Description
Currently, the research in NLP has been focusing on dealing with types of multilingual content. Thus, the first thing that we need to learn for working on different NLP tasks, such as Question Answering, is to identify the languages accurately on texts. This repository implements the idea behind the paper [A Neural Model for Language Identification in Code-Switched Tweets](https://homes.cs.washington.edu/~nasmith/papers/jaech+mulcaire+hathi+ostendorf+smith.lics16.pdf)

## Data

http://www.care4lang.seas.gwu.edu/cs2/call.html

This data is a collection of tweets; in particular,three files for the training set and three for the validation set:


* `offsets_mod.tsv`:
```
tweet_id, user_id, start, end, gold label
```

* `tweets.tsv`:
```
tweet_id, user_id, tweet text
```

* `data.tsv`:
```
tweet_id, user_id, start, end, token, gold label
```

The gold labels can be one of three:

* en
* es
* other

### Data Analysis

* As it can be seen in the following table, data are imbalanced in both the training and test set. While the number of `English` tokens in training data is about 50%, the number of `Spanish` tokens prevails in the test set.

    | label | train | dev |
    | --- | --- | --- |
    | `en` | **46042** | 3028 |
    | `es` | 25563 | **4185** |
    | `other` | 20257 | 2370 |
    | sum | 91862 | 9583 |

* The number of tweets in the training set is `7400` and in the test set is `832`. The tweets in both sets are wholly from two disjoint groups. The training set includes tweets of 6 persons and the test set has 8 persons' tweets.

    | user id | train | dev |
    | :---: | :---: | :---: |
    | 1 | 1160520883 | 156036283 |
    | 2 | 1520815188 | 21327323 |
    | 3 | 1651154684 | 270181505 |
    | 4 | 169403434 | 28261811 |
    | 5 | 304724626 | 364263613 |
    | 6 | 336199483 | 382890691 |
    | 7 |  | 418322879 |
    | 8 |  | 76523773 |
* distribution of unique tokens and characters.
    | | unique token | unique token (lower case) | unique characters |
    | :---: | :---: | :---: | :---: |
    | train | 14366 | 12220 | 50 |
    | dev | 2771 | 2559 | 28
* The distribution of the length of the tokens are depicted below which are taken by the following linux command:
    ```bash
    cut -f5 train_data.tsv|awk '{print length}'|sort -n |uniq -c|awk -F" " '{print $NF" " $(NF-1)}'|R --slave -e 'x <- scan(file="stdin", quiet=TRUE,  what=list(numeric(), numeric())); png("Histogram of tokens length-train.png");plot(x[[1]],x[[2]], xlab="length", ylab="frequency", main="Train");'
    ```
    <img src="./images/Histogram%20of%20token%20length-train.png" alt="token length distribution in training set" width="45%"/>
    <img src="./images/Histogram%20of%20token%20length-dev.png" alt="token length distribution in dev set" width="45%"/>

    It is evident that both data sets have the same distribution of tokens' lengths with a slight shift. There are several outliers in both datasets as users tend to repeat the characters on social media. The weighted average tokens' lengths for the training and test sets are `3.93` and `4.11`, respectively. I've used the following to compute these numbers:
    ```bash
    cut -f5 ../data/train_data.tsv|awk '{print length}'|sort -n |uniq -c|awk -F" " '{print $NF" " $(NF-1)}'|tr " " "*"|paste -sd+|bc -l
    ```

### Preprocssing
* Some rows in `[train|dev]_data.csv` include `"` resulting weird issue with `pandas.read_csv`. Actually, it reads the next lines till reaches another `"`, so I set `quotechar` option to `'\0'`(=NULL) in `pandas.read_csv` to solve this issue.
* I've also checked the availability of the Null in those files with the following command:
    ```bash
    grep -Pa '\x00' data/train_data.tsv
    grep -Pa '\x00' data/dev_data.tsv
    ```
* Another solution to the previous issue is the `quoting` option with `3` as its value which means `QUOTE_NONE`.
* As it is mentioned in the paper, the data contains many long and repetitive character sequences such as “hahahaha...”. To deal with these, we restricted any sequence of repeating characters to at most five repetitions with a maximum length of 20 for each token.
    ```python
    df['token'] = df['token'].apply(lambda t: re.sub(r'(.)\1{4,}',r'\1\1\1\1', t)[:20])
    ```

## Installing dependencies

You can use the `pip` program to install the dependencies on your own. They are all listed in the `requirements.txt` file.

To use this method, you would proceed as:

```pip install -r requirements.txt```

## Model
![Char2Vec](./images/Char2Vec.png)
```python
BiLSTMtagger(
(word_embeddings): Char2Vec(
    (embeds): Embedding(298, 9)
    (conv1): Sequential(
    (0): Conv1d(9, 12, kernel_size=(3,), stride=(1,))
    (1): ReLU()
    (2): Dropout(p=0.1, inplace=False)
    )
    (conv2): Sequential(
    (0): Conv1d(12, 15, kernel_size=(3,), stride=(1,))
    (1): ReLU()
    )
    (linear): Sequential(
    (0): Linear(in_features=15, out_features=15, bias=True)
    (1): ReLU()
    )
)
(lstm): LSTM(15, 128, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
(hidden2tag): Linear(in_features=256, out_features=4, bias=True)
)
```