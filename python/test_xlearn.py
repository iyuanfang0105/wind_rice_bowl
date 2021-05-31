import os
import xlearn as xl


# Training task
ffm_model = xl.create_ffm()                # Use field-aware factorization machine (ffm)
ffm_model.setTrain("../data/movie_lens_100k/train.txt")    # Set the path of training dataset
ffm_model.setValidate("../data/movie_lens_100k/test.txt")  # Set the path of validation dataset

# Parameters:
#  0. task: binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: accuracy
param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.002, 'metric': 'auc'}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, 'model.out')

# Prediction task
ffm_model.setTest("../data/movie_lens_100k/test.txt")  # Set the path of test dataset
ffm_model.setSigmoid()                 # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
res = ffm_model.predict("model.out")

print()
