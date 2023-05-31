# finger-signal-processing-sEMG

### file structure:
***
**checkpoints-cnn** : CNN 的 checkpoint

**datasets** : 已經 denoise 過的train, val data

    - dataset2 : 我第一次蒐集的 single finger data 兩組
    - dataset3 : 我第二次蒐集的 single finger data 兩組
    - dataset4 : 我第二次蒐集的 multiple finger data 三組

**raw_data** : 處理前的 raw data (目前剩你第一次搜的資料沒有被用到)

**test_datasets** : 已經 denoise 過的test data

    - testdata1 : 我第二次蒐集的不同姿勢 data 
    - testdata2 : 妳第二次蒐集的 single finger data 
    - testdata3 : 妳第二次蒐集的 multiple finger data 三組（沒有rest的所以我把testdata2的複製過來）


filter_to_csv.py : 我拿學長的code來改的，主要就是把raw data denoise 再存成 .csv (我格式是參照他sEMG-NN裡面的csv格式)

pretest.ipynb : 學長給的，裡面有多一些 visualization 的 function

**sEMG-CNN2.ipynb** : 搭配utilities2，主要是訓練在第一次的資料（超級不穩定）

**sEMG-CNN3.ipynb** : 搭配utilities3，主要是多了normalize，目前是訓練在第二次單一手指（準確度較高，但也是不穩）

**sEMG-rf.ipynb** : 把原本的 random forest 加上 testing，目前放的是用我的多手指資料 train 和 validate

**utilities_rf.py** : 整理自 utilities3.py，更正 one-hot、rest channel 自己也扣掉、加 read_testing_data() 讀測試資料

***
**TODO**

做出能比較robust的多手指模型
