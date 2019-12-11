# AIT726 Term Project
## Author: Paul Barry and Rahul Pandey
---

Code for our `Semantic Textual Similarity` project.

Dataset used: [STS Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)

Given `correlation.pl` perl file inside data to evaluate the results

All results output to replicate reports results. The evaluation metric is Pearson Correlation Coefficient (PRR)

Baseline:

a. Word2Vec (0.6235 PRR)
  * > perl data/correlation.pl data/sts-test.csv output/base_word2vec_best.txt

b. GloVe 42B (0.2987 PRR)
  * > perl data/correlation.pl data/sts-test.csv output/base_glove_42b_best.txt

c. GloVe 840B (0.5473 PRR)
  * > perl data/correlation.pl data/sts-test.csv output/base_glove_840b_best.txt

Neural Network and Logistic Regression

a. Logistic Regression (0.2778 PRR)
  * > perl data/correlation.pl data/sts-test.csv output/lr_word2vec_best.txt

b. Neural Network (0.1784 PRR)
  * > perl data/correlation.pl data/sts-test.csv output/nn_word2vec_best.txt

Language Model

a. BERT base (0.8118 PRR)
  * > perl data/correlation.pl data/sts-test.csv output/bert_base_8118.txt

b. BERT large (0.8486 PRR)
  * > perl data/correlation.pl data/sts-test.csv output/bert_large_8486.txt

c. XLNet (0.8958 PRR)
  * > perl data/correlation.pl data/sts-test.csv output/xlnet_8958_Relu.txt

NOTE: Before executing this script, please update the pretrained word embeddings file path in `config_baseline.yml`
This script requires Py YAML package. To install with pip:
> pip install pyyaml

We will go through each file to get all the approaches used:


#### baseline.py
Baseline approach to evaluate similarity based on the pretrained word embeddings.

NOTE: Before executing this script, please update the pretrained word embeddings file path in `config_baseline.yml`
This script requires Py YAML package. To install with pip:
> pip install pyyaml

Usage: python baseline.py --path <path_of_sts_data> --eval_data_type <test/train/dev> --word_embedding <word2vec/glove_42b/glove_840b>

e.g.
To evaluate test data:
> python baseline.py --path ./data/ --eval_data_type test --word_embedding word2vec
> python baseline.py --path ./data/ --eval_data_type test --word_embedding glove_42b
> python baseline.py --path ./data/ --eval_data_type test --word_embedding glove_840b

To evaluate train data:
> python baseline.py --path ./data/ --eval_data_type train --word_embedding word2vec
> python baseline.py --path ./data/ --eval_data_type train --word_embedding glove_42b
> python baseline.py --path ./data/ --eval_data_type train --word_embedding glove_840b

To evaluate dev data:
> python baseline.py --path ./data/ --eval_data_type dev --word_embedding word2vec
> python baseline.py --path ./data/ --eval_data_type dev --word_embedding glove_42b
> python baseline.py --path ./data/ --eval_data_type dev --word_embedding glove_840b

Best pearson correlation coefficient got was 62.35% with word2vec on test set.

Flow:
i. main
ii. Parse arguments
iii. Load dataset  (load_data)
    1. Open and preprocess train, validation, and test datasets.
iv. Get baseline results (get_baseline_results_embeddings)
    1. Load the pretrained word embeddings
    2. for each sentences, get the average of their word embeddings vectors
    3. Take cosine similarity and scale to 5
v. store and evaluate the results (evaluate_result)


#### Logistic Regression and Feed Forward Neural Network Approach
Neural Network and Logistic Regression approach to train and evaluate similarity initialized with pretrained word embeddings.

NOTE: Before executing this script, please update the pretrained word embeddings file path in `config_baseline.yml`
This script requires Py YAML package. To install with pip:
> pip install pyyaml

Usage: python nn_base.py --path <path_of_sts_data> --eval_data_type <test/train/dev> --word_embedding <word2vec/glove_42b/glove_840b> --model <lr/nn>

e.g.
To evaluate test data:
> python nn_base.py --path ./data/ --eval_data_type test --word_embedding word2vec --model lr

Best pearson correlation coefficient got was 27.78% with logistic regression and word2vec on test set.

Flow:
i. main
ii. Parse arguments
iii. Load dataset  (load_data)
    1. Open and preprocess train, validation, and test datasets.
iv. load word embeddings
v. Initialize the model (Logistic Regression/Neural Network) (LogReg/FFNN)
vi. Train the model
vii. Predict and evaluate on test data

#### Language Model
#### BERT based Approach
Usage: python BERT_STS.py
Arguments:
    --train_set for training data file
    --valid_set for validation data file
    --test_set for test data file
    --learning_rate to set the learning rate of the file
    --output_file to set the path and filename for the model's predictions
    --early_stopping to set the early stopping threshold
Best pearson correlation coefficient uses default parameters to achieve 84.86.

Flow:
i. main
ii. Parse arguments
iii. Generate dataset
    1. Open and preprocess train, validation, and test datasets.
    2. Stored in StsDataset objects
v. Train the model
    1. Load pretrained BERT weights and initialize FFNN weights
    2. Create Loss and Optimization objects
    3. for each epoch
        a. For each batch
            I. calculate y_hat from forward pass
            II. Calculate loss from y_hat to y
            III. Update the weigths with gradient times learning rate
        b. Do forward pass with validation set.
        c. Calculate loss on validation set.
        d. If new minimum loss on validation set, then save weights to RAM.
        e. If early stopping threshold hit without improvement on validation set, then end training.
vi. Test the model
    1. Forward pass through test dataset.
    2. Write prediction to output file.


#### XLNet based Approach
Usage: python self_attention_STS.py
Arguments:
    --train_set for training data file
    --valid_set for validation data file
    --test_set for test data file
    --learning_rate to set the learning rate of the file
    --output_file to set the path and filename for the model's predictions
    --early_stopping to set the early stopping threshold
Best pearson correlation coefficient uses default parameters to achieve 89.58.

Flow:
i. main
ii. Parse arguments
iii. Generate dataset
    1. Open and preprocess train, validation, and test datasets.
    2. Stored in StsDataset objects
v. Train the model
    1. Load pretrained BERT weights and initialize FFNN weights
    2. Create Loss and Optimization objects
    3. for each epoch
        a. For each batch
            I. calculate y_hat from forward pass
            II. Calculate loss from y_hat to y
            III. Update the weigths with gradient times learning rate
        b. Do forward pass with validation set.
        c. Calculate loss on validation set.
        d. If new minimum loss on validation set, then save weights to RAM.
        e. If early stopping threshold hit without improvement on validation set, then end training.
vi. Test the model
    1. Forward pass through test dataset.
    2. Write prediction to output file.
