# Measuring Accuracy and Consistency of Large Language Models for OOAD Principles
A final research project for CSCI 5448 - Object-Oriented Analysis and Design where we measure accuracy and consistency of Large Language Models for SOLID principles

Students:
* **Federico Ortega Riba**
* **Sheetal Sharma**

---

# DEPENDENCIES 

# 1. Setting up a virtual environment

### Initial Considerations

* Set up your coding environment (if it isn't already)
* Install your IDE of choice — I like VS Code *Install Conda (or equivalent Python package manager)
* Install Git
* Configure Git
* Set up the environment locally: clone this repo

### Python virtual env configuration

```src
conda create -n ooad-acc
conda activate ooad-acc
```

### Install REQUIREMENTS (you only need to do this once)

```src
pip install -r requirements.txt
```

### Getting an API Key

For this framework, we will use an open LLM: Llama-4-Maverick-17B-128E-Instruct-FP8

In order to obtain your API key, please visit the [llama.developer.meta.com](https://llama.developer.meta.com/). Take into account that API Keys can take some days to generate.
Once you have this, fill the .env file with your API key in quotes.

You should be all set to evaluate your code!

# BUILD INSTRUCTIONS

# 2. How to run each module from the CLI

### solid_accuracy_evaluator_strings_only.py

If our code is just a single string, we can prompt Llama using this module, following this structure:

```src
python solid_accuracy_evaluator_strings_only.py --text "<your code here>" --pretty
```

### solid_accuracy_evaluator_lists_strings.py

If our code format is more complicated, we have two options:

1. For a list of code like '["code A","code B"]', we do the following:

```src
python solid_accuracy_evaluator_lists_strings.py --text '["code A","code B"] --pretty
```

2. For a .txt file with all our code in the format '["code A","code B"]', we do the following:

```src
python solid_accuracy_evaluator_lists_strings.py --from-file your_txt_file.txt --pretty
````

Both of these approaches will return a JSON-like output that we can copy in a .txt file for further evaluation. 

⚠ It is very important not to lose track of the code snippets in order to make sure that we are comparing our annotations to the LLM-generated annotations. ⚠

# 3. Pre-processing our Dataset

Our work builds upon the [Pehlivan et al. dataset](https://zenodo.org/records/17513921) from the research paper [_Are We SOLID Yet? An Empirical Study on Prompting LLMs to Detect Design Principle Violations_](https://www.arxiv.org/abs/2509.03093).

Since this dataset is formatted as a JSON file, we have made the extrac_json_values module to match our annotations to the output format of the LLM.

In case you also have an annotated JSON file that needs preprocessing, you can run the following:

```src
python extract_json_values.py --in your_dataset.json --out your_outpu_file.txt --pretty
```

This will create a .txt file in your directory that now can be compared with your LLM responses. In our case, it is **human_annotations_results**, you can also work with that as an example.

# 4. Evaluating Accuracy of LLM's responses

Now that we have two .txt files (human_annotations_results.txt & LLM_results.txt) we are ready to evaluate. 

We will use Spearman and Pearson correlation, which work well for this Likert scale task of giving a score between 1-5. We can conduct an evaluation on two variables: adherence_score and violation_severity

To evaluate adherence score, run the following code:
```src
python spearman_pearson_correlation_from_txt.py \
  --file-a human_annotations.txt \
  --file-b llm_annotations.txt \
  --metric adherence_score \
  --out-fig agreement_adherence.png \
  --pretty
````

To evaluate violation_severity, run the following code:
```src
python spearman_pearson_correlation_from_txt.py \
  --file-a human_annotations.txt \
  --file-b llm_annotations.txt \
  --metric violation_severity \
  --out-fig agreement_severity.png \
  --pretty
```

You will get a figure with the results in a matrix format. Bear in mind that you will need a moderate number of annotations to measure how well it captures accuracy.

# 5. Evaluating Consistency of LLM's Labels

For this, we will need a txt. with our error in a dictionary where "input" is the key and the code snippet is the value.

```src
python consistency_label_eval.py \
  --dataset snippets_example.txt \
  --k [number of runs] \
  --pretty
```

The output is a JSON array of objects like {"index": 0, "labels": [...], "label_agreement": 0.96} that is used by the downstream scripts.

Now, if we want to get the average label-agreement, we have to paste all the evaluations in a new .txt file and run the following code:

```src
python label_agreement_stats.py consistency_k5.txt
```

For the evaluation of accuracy, precision, recall and F1 score between gold standard and LLM's annotations, run the following:

```
python3 ooad_label_evaluator.py \
  gold_standard_labels.txt \
  consistency_k5.txt \
  confusion_matrix_k5.png
```

# TEST INSTRUCTIONS

All tests were created with Pytest. Instructions on how to run them depend on IDE of choice.

In VSCode, we can go to the last icon on the left and run all of them as in the following screenshot:

![alt text](test_coverage.png)

Bear in mind that tests are only to test the architecture, we don't test using the API Key. 

Test coverage is the following:

![alt text](test_coverage.png)



