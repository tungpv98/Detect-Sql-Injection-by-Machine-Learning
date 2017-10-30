# predict_stock_RNN.py
This python code is predict stock with LSTM(Long-Short Term Memory layer).
<br>Not perfect but, it seems to be cool. Maybe boltzmann machine is batter.
<br>You can understand how to handle data with [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).
<br>If you have a good computing power(nice CPU or GPU) then add more LSTM layer and increase epochs.
<br>Maybe someone send e-mail to me with more upgraded model.
<br>Thanks :D
<br>
<br>**Result**
```
train_data: 1379
test_data: 592

Epoch 1/100
21s - loss: 0.0028
Epoch 2/100
16s - loss: 9.8225e-04
Epoch 3/100
16s - loss: 9.7002e-04
Epoch 4/100
17s - loss: 9.4444e-04
Epoch 5/100
...
...
...
...
Epoch 95/100
16s - loss: 2.8667e-04
Epoch 96/100
17s - loss: 3.2657e-04
Epoch 97/100
16s - loss: 3.0850e-04
Epoch 98/100
16s - loss: 2.6505e-04
Epoch 99/100
18s - loss: 3.0946e-04
Epoch 100/100
20s - loss: 2.8936e-04

Train score: 7.99 RMSE
Test score: 15.48 RMSE
```
<br>![Google Stock Predict Result](https://github.com/Scott-Park/MachineLearning/blob/master/DeepLearning/RNN/result_of_google_stock_prediction(LSTM).png)
