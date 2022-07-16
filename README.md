# PyTorch-Detect-Code-Switching

## Task Description
Currently, the research in NLP has been focusing on dealing with types of multilingual content. Thus, the first thing that we need to learn for working on different NLP tasks, such as Question Answering, is to identify the languages accurately on texts.

### Required reading

[1] https://homes.cs.washington.edu/~nasmith/papers/jaech+mulcaire+hathi+ostendorf+smith.lics16.pdf \
[2] http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

## Data

http://www.care4lang.seas.gwu.edu/cs2/call.html

This data is a collection of tweets; in particular,three files for the training set and three for the validation set:


* `offsets_mod.tsv`:
```
tweet_id, user_id, start, end, gold label
```

* `tweets.tsv`
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

### Data Distribution
As it can be seen in the following table, data are imbalanced in both the training and test set. While the number of `English` tokens in training data is about 50%, the number of `Spanish` tokens prevails in the test set.
| label | train | dev |
| --- | --- | --- |
| `en` | **46042** | 3028 |
| `es` | 25563 | **4185** |
| `other` | 20257 | 2370 |
| sum | 91862 | 9583 |

The number of tweets in the training set is `7400` and in the test set is `832`. The tweets in both sets are wholly from two disjoint groups. The training set includes tweets of 6 persons and the test set has 8 persons' tweets.
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

### Preprocssing
* Some rows in `[train|dev]_data.csv` include `"` resulting weird issue with `pandas.read_csv`. Actually, it reads the next lines till reaches another `"`, so I set `quotechar` option to `'\0'`(=NULL) in `pandas.read_csv` to solve this issue.
* I've also checked the availability of the Null in those files with the following command:
    ```bash
    grep -Pa '\x00' data/train_data.tsv
    grep -Pa '\x00' data/dev_data.tsv
    ```
* Another solution to the previous issue is the `quoting` option with `3` as its value which means `QUOTE_NONE`.

## Installing dependencies

You can use the `pip` program to install the dependencies on your own. They are all listed in the `requirements.txt` file.

To use this method, you would proceed as:

```pip install -r requirements.txt```