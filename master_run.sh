python /home/hazim/Desktop/Categorical-Feature-Encoding-Challenge/src/create_folds.py

python /home/hazim/Desktop/Categorical-Feature-Encoding-Challenge/src/train.py --fold 0 --model randomforest
python /home/hazim/Desktop/Categorical-Feature-Encoding-Challenge/src/train.py --fold 1 --model randomforest
python /home/hazim/Desktop/Categorical-Feature-Encoding-Challenge/src/train.py --fold 2 --model randomforest
python /home/hazim/Desktop/Categorical-Feature-Encoding-Challenge/src/train.py --fold 3 --model randomforest
python /home/hazim/Desktop/Categorical-Feature-Encoding-Challenge/src/train.py --fold 4 --model randomforest

python /home/hazim/Desktop/Categorical-Feature-Encoding-Challenge/src/predict.py --model randomforest