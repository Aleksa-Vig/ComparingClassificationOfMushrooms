# ComparingClassificationUCI

A repo to compare and contrast classification techniques using scikit learn to compare different models on the UC [Irvine Waveform Dataset](https://archive-beta.ics.uci.edu/datasets?search=Waveform%20Database%20Generator%20(Version%202)).

Please see FAQ for questions on how to find the results...
## How to run

1. Make sure your venv for python is activiated (if on pycharm this should be automatic): \
``(venv) PS C:\..... <--present in terminal ``


2. Install the dependencies using pip \
``pip install -r requirements.txt ``


3. Run the optimized pipeline (This trains, fits, and then outputs performance statistics on the models) \
``python OptimizedPipeline.py``


4. Check the Outputs/Output/ Optimized*.csv (Any file prepended with Optimized) \
to see how each OPTIMIZED model performed and their respective statistics  
``Statistics in the csv include: hyperparameter-set, accuracy, precision, recall, f1 score, and execution times``


5. (Optional) Run the un-optimized pipeline Reg-*.csv \
``python OptimizedPipeline.py``


6. (Optional) Check the Outputs/Output/ Reg*.csv (Any file prepended with Reg) \
to see how each model UNOPTIMIZED performed and their respective statistics  
``Statistics in the csv include: hyperparameter-set, accuracy, precision, recall, f1 score, and execution times``

## FAQ

### Where can I see the results? 
Our final results can be found in ```Outputs\Output``` in the \
``OK-Optimized-Models-Scores-allModels2023-03-26AtTime16;20;57-509561.csv`` 
and ``OK-Reg-Models-Scores-allModels2023-03-26AtTime15-56-35-804754.csv`` for Optimized and Unoptimized 
performance/statistics respectively. Any new generated data can also be found in this folder with the time and date
of creation the file name.

### I can't run the repo! I am missing dependencies! 
If by chance all dependencies don't install, please use ``pip`` to manually install any missing packages to the virtual environment
