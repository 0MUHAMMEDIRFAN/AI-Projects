# Simple ML Text Classifier

This project is a simple text classification program that uses a Naive Bayes-like approach to classify sentences as either "sports" or "politics". It is trained on a small set of labeled sentences and predicts the category of new input sentences based on word frequencies in each class.

## How it works
- The program is trained on example sentences labeled as either "sports" or "politics".
- It calculates word frequencies for each class and uses these to compute the probability that a new sentence belongs to each class.
- The user can interactively enter sentences, and the program will output whether the sentence is related to sports or politics.

## Usage
1. Run `main.py` in your Python environment.
2. Enter a sentence when prompted.
3. The program will predict and print either "sports" or "politics" based on your input.

## Example
```
Enter your thoughts : The match was exciting
sports

Enter your thoughts : The parliament passed a new law
politics
```
