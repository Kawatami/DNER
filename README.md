# DNER

This repository provides a tool box for the Dynamic Named Entity Recognition framework.

### Data

Data can be found at this [link](https://drive.google.com/file/d/1u8zQZV1gHyzt3bPF4RGQSt_dciRvVKLA/view?usp=sharing), or manually search 
for :
```https://drive.google.com/file/d/1u8zQZV1gHyzt3bPF4RGQSt_dciRvVKLA/view?usp=sharing```

Data should be placed in the data\ directory to use pre defined scripts.

Three environment variable soudle be set : 
+ `$DATA_ROOT` : Directory holding the data (the directory `DNER_datasets` should be placed under it)
+ `$LOG_ROOT` : Directory holding the logged data (mainly `tensorboard` files)
+ `$INFERENCE_ROOT` : Directory holding the model output data
### Files

The repo is organized as follow : 
+ run.py :  main python script for training models
+ script\ : contains sh script with predefined configuration for model training
+ source\ : source files
+ data\ : directory holding training data 
+ requirement.txt : requirements file for module importation