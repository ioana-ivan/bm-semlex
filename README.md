# bm-semlex 

This project provides the tools to create an evaluation test suite for language models from resources such as Semcor and Wordnet.
More precisely, the tools enable the creation of tests for synonymy in context.

They enable the following process:
1) extract quadruples of target word, synonym, other synonym and target word in context (example containing the target word) from the above-mentioned resources
2) from these quadruples, construct tests for synonymy having different forms
3) evaluate language models
4) compute results
5) analyze results


## Installation
A requirements file will be made available in the root of the project. \
Python 3.9.19.

Install the testing package with the following command in the project root: \

```
pip install --editable .
```

Now you should be able tu run any script in the /src folder.

## File structure

Two main folders, **src** for code and **output** for data. \

- output
    - synonyms
        - negative : File containing nouns for random selection
        - pairs_triples_examples : Basefile used in project as basis for tests
        - result
            - eval : File containing test results (perplexity) for different tests performed on the main models
            - test : Test files (naming of test files described in test_config.yaml)
            - tokens : File containing tokens difference used in analysis
- src 
    - testing_pkg
        - analyze : Analyze results based on files from output/synonyms/result
        - lm : Evaluate LMs based on different criteria and compute results
        - test : Create tests starting from test definitions in test_config.yaml
    - training
        - utility for tokenizing
    - utils
        - processing utils
    - extract_from_semcor.py : Extract the base filename from Semcor and WordNet
    - extract_words.py : Extract the negative words from WordNet
- other_files : Instructions for synonymy evaluation for humans




## AI assistance

Copilot has been used to generate potential continuations while writing code and for factorisation.
ChatGPT has also been used occasionally for factorisation.

