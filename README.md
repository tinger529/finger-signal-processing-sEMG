# finger-signal-processing-sEMG

### file structure:
***
dataset2 : 我那天搜的資料，兩種不同的姿勢，已經denoise過

以下都已經denoise (band pass filter):

dataset3 : 我的單一手指data

dataset4 : 我的多手指data

testdata1 : 後面搜的test data

testdata2 : 你的單一手指data

testdata3 : 你的多手指data

raw1 : 你那天搜的資料，只有三根手指各兩個channel

raw2 : 我那天搜的資料

raw3 : 5/23 data

filter_to_csv.py : 我拿學長的code來改的，主要就是把raw data denoise 再存成 .csv (我格式是參照他sEMG-NN裡面的csv格式)

pretest.ipynb : 學長給的，裡面有多一些 visualization 的 function

sEMG-Neural-Net : (只有列出我覺得用得到的)

    - checkpoints-cnn : train cnn 存下來的check point
    - sEMG : 訓練資料，裡面應該只有csv是訓練需要的，mat不用
    - sEMG-CNN.ipynb : 這個已經改好了，可以train
    - sEMG-LSTM.ipynb : 這個在跑的時候我的kernel一直crash掉
    - utilities.py : 處理dataset的部分

新增的code:

    - sEMG-CNN3.ipynb: 套用了全新的utility3技術
    - utility3.py: 套用了normalize

**TODO**
可以試看看不同模型，我試試看作fft
