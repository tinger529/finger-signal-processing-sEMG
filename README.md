# finger-signal-processing-sEMG

### file structure:
***
dataset2 : 我那天搜的資料，兩種不同的姿勢，已經denoise過

raw1 : 你那天搜的資料，只有三根手指各兩個channel

raw2 : 我那天搜的資料

filter_to_csv.py : 我拿學長的code來改的，主要就是把raw data denoise 再存成 .csv (我格式是參照他sEMG-NN裡面的csv格式)

pretest.ipynb : 學長給的，裡面有多一些 visualization 的 function

sEMG-Neural-Net : (只有列出我覺得用得到的)

    - checkpoints-cnn : train cnn 存下來的check point
    - sEMG : 訓練資料，裡面應該只有csv是訓練需要的，mat不用
    - sEMG-CNN.ipynb : 這個已經改好了，可以train
    - sEMG-LSTM.ipynb : 這個在跑的時候我的kernel一直crash掉
    - utilities.py : 處理dataset的部分

**TODO**

需要先做的主要是把utilities改好