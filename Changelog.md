July 19, 2022
July 15, 2022
 * Printing the loss for train/val set on the screen
 * computation of `f1_score` for both training and validation set shows the network convergence
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

July 14, 2022
 * Data Class improvement
    * several dictionaries to convert token,label,char to id and vice versa
    * making the coded sentences and their counterparts labels
 * LSTM class
 * `CodeSwitchDataset` as well as customized DataLoader

July 13, 2022
 * github repo initailization
 * reading the paper
 * starting the code with Data Class
    * an issue with quoting in reading `tsv` files