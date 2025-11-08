# Measuring Accuracy and Consistency of Large Language Models for OOAD Principles
A final research project for CSCI 5448 - Object-Oriented Analysis and Design where we measure accuracy and consistency of Large Language Models for SOLID principles

Students:
* **Federico Ortega Riba**
* **Sheetal Sharma**

---

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

### Getting an API Key

For this framework, we will use an open LLM: Llama-4-Maverick-17B-128E-Instruct-FP8

In order to obtain your API key, please visit the [llama.developer.meta.com](https://llama.developer.meta.com/). Take into account that API Keys can take some days to generate.
Once you have this, fill the .env file with your API key in quotes.

You should be all set yo evaluate your code!

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

This will create a .txt file in your directory that now can be compared with your LLM responses.

# 4. Evaluating Accuracy of LLM's responses

Now that we have two .txt files we are ready to evaluate. We will use Spearman and Pearson correlation, which work well for this Likert scale task of giving a score between 1-5. We can conduct an evaluation on two variables: adherence_score and violation_severity

To evaluate adherence score, run the following code:
```src
python spearman_correlation_from_txt.py \
  --file-a results_a.txt \
  --file-b results_b.txt \
  --out-fig correlation.png \
  --metric adherence_score \
  --pretty
````

To evaluate violation_severity, run the following code:
```src
python spearman_correlation_from_txt.py \
  --file-a results_a.txt \
  --file-b results_b.txt \
  --metric violation_severity \
  --out-fig correlation_vs.png \
  --pretty
```

You will get a figure with the results. Bear in mind that you will need a moderate number of annotations to measure how well it captures accuracy.


