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

### Preprocssing
* Some rows in `[train|dev]_data.csv` include `"` resulting weird issue with `pandas.read_csv`. Actually, it reads the next lines till reaches another `"`, so I set `quotechar` option to `'\0'`(=NULL) in `pandas.read_csv` to solve this issue.
* I've also checked the availability of the Null in those files with the following command:
    ```bash
    grep -Pa '\x00' data/train_data.tsv
    grep -Pa '\x00' data/dev_data.tsv
    ```
* Another solution to the previous issue is the `quoting` option with `3` as its value which means `QUOTE_NONE`.