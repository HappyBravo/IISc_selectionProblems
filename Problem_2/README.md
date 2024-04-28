<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<h1 align="center">Problem 2: Data Science Challenge</h3>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#requirements">Requirements</a>
    </li>
    <li>
      <a href="#the-goal">The Goal</a>
    </li>
    <li>
      <a href="#task-1-data-structuring">Task 1</a>
    </li>
    <li>
      <a href="#task-2-text-preprocessing-and-vectorization">Task 2 </a>
    </li>
  </ol>
</details>



## Requirements

Ensure you have the following packages installed:

- `pandas` : For data manipulation and analysis.
- `numpy` : For numerical operations and array manipulation.
- `tqdm` : For displaying progress bars during data processing.
- `torch` : For building and training neural networks.
- `transformers` : For accessing pre-trained language models like BERT.
- `csv`: For reading and writing CSV files.
- `json`: For reading and writing JSON files.
- `os`: For operating system-related functionality.


<!-- ABOUT THE PROJECT -->
## The Goal

This project addresses two key tasks related to Natural Language Processing (NLP) classification techniques.

- Task 1: Data Structuring
  -  Read data files and their content from a compressed zip folder.
  - The project extracts data from a zip archive and transforms it into a structured CSV format following a predefined schema.

- Task 2: Text Preprocessing and Vectorization
  - Prepare textual data for use in NLP classification models.
  - The program reads the CSV file generated in Task 1 and perform any of preprocessing steps like tokenization, lowercasing, preprocessing.
  - The ultimate objective is to convert the preprocessed text data into numerical representations (vectors) suitable for training NLP classification models.

  > (We are free to choose how we vectorize the data to get features)

---

## Task 1: Data Structuring

- We read data from its source and transformed it into a predefined structured format (CSV) with specified schema. 
- We stored it in [bbc_Articles.csv](./bbc_articles.csv) file.

## Task 2: Text Preprocessing and Vectorization

- We load the [CSV file](./bbc_articles.csv) generated in Task 1.

- One-hot encoding is applied to the `category` column within the loaded dataset.

- Text Vectorization with Large Language Models (LLMs)
  - Depending on the specific LLM in use, we segmented texts into chunks of the desired length. This is necessary as texts exceeding the model’s maximum token limit risk truncation, potentially leading to the loss of vital information. Splitting the text into smaller segments mitigates this risk and ensures the preservation of crucial content.
  - We employed LLM to handle the entire text vectorization pipeline, encompassing tokenization, embedding generation, and (due to LLM capabilities) lowercasing (which becomes unnecessary).
  - Utilizing an LLM offers several advantages:
    - LLM manages all stages within a single library, reducing code complexity.
    - LLMs are trained on massive datasets because of which are adept at generating high-quality embeddings that capture intricate semantic relationships within text data. This has the potential to enhance performance in NLP tasks.
- The generated vector representations/embeddings are stored within a newly created column named  `<model_name>_embeddings`.
- Finally, we created a new [CSV file](./vectorized_dataset.csv). It contains both the original text data and its corresponding embeddings. This gives flexibility of utilizing either the text or its embeddings based on specific project requirements.

---
---

<h3 align="center"> - x - X - x -</h3>
