# Public Perspectives on Myalgic Encephalomyelitis/Chronic Fatigue Syndrome: A Twitter Thematic and Sentiment Analysis
This is the repository that contains the raw, unprocessed tweets and the analysis codes for the project. The environment can be replicated by installing *requirements-windows.txt* and initializing the VS Code workspace using *twitter_co_sa.code-workspace*. Please note that if you wish to acquire any processed datasets or results, please contact the corresponding author Dr. Jason Busse at bussejw@mcmaster.ca, and we will accommodate any reasonable requests.

## analysis.ipynb
Contains ALL necessary code to replicate the findings. All section numberings (e.g., 1.3) refers to code within this file. 

### get_shap_scores.py
This contains the function used to combine SHAP scores from chunked datasets in 4.3. This function is stored separately because it is the worker function for multiprocessing. Multiprocessing was used to drastically speed up this process as the SHAP library has poor efficiency slicing into SHAP Explanation objects.

## shapley.py and shapley.sh
These two files are used to submit the necessary jobs to the Digital Research Alliance of Canada to calculate SHAP scores in parallel. If a similar high performance computing cluster is not available, code in 4.2 can also be used. Both logits or probabilities can be used to calculate the SHAP scores.

## datasets/
Contains all raw and processed datasets used for analysis. The 3 cleaned datasets are also stored directly under this directory.

### datasets/raw/
Contains the 7z file of the 3 raw datasets that contained all tweets used for analysis.

### datasets/chunked/
Contains 92 (0-91) chunked datasets where each chunk contains at most 10,000 tweets. These datasets need to be generated using code from section 4.2. Originally these were calculated in parallel using computing power from the Digital Research Alliance of Canada. If only 1 GPU is available, expect approximately 2 months for the calculation of all SHAP values depending on the processing power available.

## lda_figures/
After 5.2, all resulting figures and the description csv will be in this directory.

## lda_results/
Contains the .pkl files of the LDA models when 5.1 is ran.

## sentiment_results/
Contains .csv files of the sentiment probabilities and classifications of all tweets from RoBERTa (2.1) or Bing Liu Lexicon (2.2).

## shap_figures/
Contains all SHAP figures from 4.4.

## shap_results/
Contains all .pkl files of SHAP Explanation objects from each dataset chunk (from 4.2). After 4.3, a csv file will be generated, containing the SHAP values for all features from all tweets. This file will be approximately 10GB with ~90 million rows.

## stat_results/
Contains csv tables from step 3. These tables are displayed in the notebook regardless.