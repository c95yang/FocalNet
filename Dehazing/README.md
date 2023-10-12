### Download the Datasets
- reside-indoor [[gdrive](https://drive.google.com/drive/folders/1pbtfTp29j7Ip-mRzDpMpyopCfXd-ZJhC?usp=sharing), [Baidu](https://pan.baidu.com/s/1jD-TU0wdtSoEb4ki-Cut2A?pwd=1lr0)]
- reside-outdoor [[gdrive](https://drive.google.com/drive/folders/1eL4Qs-WNj7PzsKwDRsgUEzmysdjkRs22?usp=sharing)]
- (Separate SOTS test set if needed) [[gdrive](https://drive.google.com/file/d/16j2dwVIa9q_0RtpIXMzhu-7Q6dwz_D1N/view?usp=sharing), [Baidu](https://pan.baidu.com/s/1R6qWri7sG1hC_Ifj-H6DOQ?pwd=o5sk)]
### Train on RESIDE-Indoor

~~~
cd ITS
python main.py --mode train --data_dir your_path/reside-indoor
~~~


### Train on RESIDE-Outdoor
~~~
cd OTS
python main.py --mode train --data_dir your_path/reside-outdoor
~~~


### Evaluation
#### Download the model [here](https://drive.google.com/drive/folders/1HXCwpDbzRL9KLc9XPhUPf2YisS_1wDxo?usp=sharing)
#### Testing on SOTS-Indoor
~~~
cd ITS
python main.py --data_dir your_path/reside-indoor --test_model path_to_its_model
~~~
#### Testing on SOTS-Outdoor
~~~
cd OTS
python main.py --data_dir your_path/reside-outdoor --test_model path_to_ots_model
~~~

For training and testing, your directory structure should look like this

`Your path` 
 `├──reside-indoor` 
     `├──train`  
          `├──gt`  
          `└──hazy`  
     `└──test`  
          `├──gt`  
          `└──hazy`  
 `└──reside-outdoor` 
     `├──train`  
          `├──gt`  
          `└──hazy`  
     `└──test`  
          `├──gt`  
          `└──hazy` 