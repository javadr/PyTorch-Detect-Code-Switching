TODO:
 * running on the CoLab with GPU activated.
 * fine tunning to find the best parameters
 * Most of the time the predicition of the first token is not correct.
 * Issue with the last token; seen as <PAD>!, solved awkwardly be adding another extra token.
 * Using mask in computation of loss

===================================================================== July 18, 2022
 * `predict.py` for the ordinary application.
 ```python
  python predict.py --text [sample text] --model [pretrained model]
 ```
 * Since I've used F1_Score with micro average, I should mention here that the it means
 `micro-F1 = micro-precision = micro-recall = accuracy`, i.e. I've reported in all cases `accuray` not `f1-score`.
 From now on, I use `macro` average for `f1-score`. Therefore, the result would be more realistic now.

![plot](./images/plot[2207181658]-Ep14B64BiLSTM+Char2Vec,%202Layers,%20Adam,%20lre-3,%20wde-5.png)

 * Saving the best model for prediction
 * multiple-width filter bank in the second layer of thc Char2Vec --> better result and less overfitting.

```python
BiLSTMtagger(
  (word_embeddings): Char2Vec(
    (embeds): Embedding(298, 9)
    (conv1): Sequential(
      (0): Conv1d(9, 12, kernel_size=(3,), stride=(1,))
      (1): ReLU()
      (2): Dropout(p=0.25, inplace=False)
    )
    (convs2): ModuleList(
      (0): Sequential(
        (0): Conv1d(12, 5, kernel_size=(3,), stride=(1,))
        (1): ReLU()
      )
      (1): Sequential(
        (0): Conv1d(12, 5, kernel_size=(4,), stride=(1,))
        (1): ReLU()
      )
      (2): Sequential(
        (0): Conv1d(12, 5, kernel_size=(5,), stride=(1,))
        (1): ReLU()
      )
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

![plot](./images/plot[2207181333]-Ep14B64BiLSTM+Char2Vec,%202Layers,%20Adam,%20lre-3,%20wde-5.png)

===================================================================== July 17, 2022

 * F1 score with `weighted` average instead of `micro`.
 * Char2Vec class
 * removing repetition in a token with more than 4 characters and truncation of any words to the length of at most 20 characters; ==> a slightly better result
 * Char2Vec+BiLSTM finished, with f1=0.9549, val_f1=0.9443; another slight improvement in the model
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

![plot](./images/plot[2207171959]-Ep40B64BiLSTM+Char2Vec,%202Layers,%20Adam,%20lre-3,%20wde-5.png)


===================================================================== July 16, 2022

 * dechipher the text/label from the output of network

 * tokens should be considered in the context, not as a collection of single tokens:
 in the following `audio` is a Spanish token, not an English one.
   > @andres_romero17 si , prometo hacer un audio :)

   > other es other es es es es other
 * loss/f1_score plot

 ![plot](./images/plot[2207161342]-Ep40BiLSTM,%202Layers,%20Adam,%20lre-3,%20wde-5.png)

 * data analysis around tweets and their tokens/chars
 * code sanitization


===================================================================== July 15, 2022

 * Printing the loss for train/val set on the screen
 * computation of `f1_score` for both training and validation set shows the network convergence
 * SGD, lr=0.1, hidden_dim=64
 ```bash
Epoch  1/40, loss=0.9072, val_loss=0.8901    ,train_f1=0.5998, val_f1=0.5462
Epoch  2/40, loss=0.6987, val_loss=0.7863    ,train_f1=0.7165, val_f1=0.6602
Epoch  3/40, loss=0.5788, val_loss=0.7573    ,train_f1=0.7714, val_f1=0.7342
Epoch  4/40, loss=0.4912, val_loss=0.7454    ,train_f1=0.8088, val_f1=0.7589
Epoch  5/40, loss=0.4221, val_loss=0.7322    ,train_f1=0.8367, val_f1=0.7747
Epoch 10/40, loss=0.2226, val_loss=0.6976    ,train_f1=0.9123, val_f1=0.7897
Epoch 15/40, loss=0.1427, val_loss=0.7406    ,train_f1=0.9431, val_f1=0.8072
Epoch 20/40, loss=0.1083, val_loss=0.6276    ,train_f1=0.9577, val_f1=0.8133
Epoch 25/40, loss=0.0925, val_loss=0.6425    ,train_f1=0.9648, val_f1=0.8163
Epoch 30/40, loss=0.0842, val_loss=0.6611    ,train_f1=0.9683, val_f1=0.8171
Epoch 35/40, loss=0.0792, val_loss=0.6735    ,train_f1=0.9701, val_f1=0.8178
Epoch 40/40, loss=0.0763, val_loss=0.6753    ,train_f1=0.9711, val_f1=0.8180
   ```
 * Adam+ReduceLROnPlateau, lr=1e-3, wd=1e-5, hidden_dim=128
 ```bash
Epoch 1/7, loss=0.5991, val_loss=0.5572    ,train_f1=0.7311, val_f1=0.7483
Epoch 2/7, loss=0.2947, val_loss=0.4787    ,train_f1=0.9005, val_f1=0.8266
Epoch 3/7, loss=0.1783, val_loss=0.4336    ,train_f1=0.9485, val_f1=0.8379
Epoch 4/7, loss=0.1256, val_loss=0.4124    ,train_f1=0.9653, val_f1=0.8494
Epoch 5/7, loss=0.1049, val_loss=0.3998    ,train_f1=0.9698, val_f1=0.8512
Epoch 6/7, loss=0.0977, val_loss=0.3884    ,train_f1=0.9714, val_f1=0.8512
Epoch 7/7, loss=0.0940, val_loss=0.3817    ,train_f1=0.9725, val_f1=0.8529
 ```
 * Minibatches made a great leap: train_f1=0.97, val_f1=0.94
 ```bash
Epoch  1/40, loss=0.5998, val_loss=0.4027    ,train_f1=0.7502, val_f1=0.7768
Epoch  2/40, loss=0.3764, val_loss=0.3790    ,train_f1=0.8179, val_f1=0.7971
Epoch  3/40, loss=0.3242, val_loss=0.3561    ,train_f1=0.8501, val_f1=0.8307
Epoch  4/40, loss=0.2618, val_loss=0.2922    ,train_f1=0.8861, val_f1=0.8741
Epoch  5/40, loss=0.2209, val_loss=0.2553    ,train_f1=0.9065, val_f1=0.8931
Epoch 10/40, loss=0.1291, val_loss=0.1723    ,train_f1=0.9460, val_f1=0.9291
Epoch 15/40, loss=0.0892, val_loss=0.1429    ,train_f1=0.9616, val_f1=0.9419
Epoch 20/40, loss=0.0665, val_loss=0.1471    ,train_f1=0.9675, val_f1=0.9409
Epoch 25/40, loss=0.0510, val_loss=0.1481    ,train_f1=0.9715, val_f1=0.9397
Epoch 30/40, loss=0.0420, val_loss=0.1676    ,train_f1=0.9742, val_f1=0.9397
Epoch 35/40, loss=0.0359, val_loss=0.1756    ,train_f1=0.9755, val_f1=0.9386
Epoch 40/40, loss=0.0323, val_loss=0.1934    ,train_f1=0.9765, val_f1=0.9403
 ```
 * BiLSTM with 2 layers and dropout prevents kind of overfitting

===================================================================== July 14, 2022
 * Data Class improvement
    * several dictionaries to convert token,label,char to id and vice versa
    * making the coded sentences and their counterparts labels
 * LSTM class
 * `CodeSwitchDataset` as well as customized DataLoader

===================================================================== July 13, 2022
 * github repo initailization
 * reading the paper
 * starting the code with Data Class
    * an issue with quoting in reading `tsv` files