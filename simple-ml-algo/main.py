# Training data
train_data = [
    ("India won the cricket match", "sports"),
    ("The prime minister gave a speech", "politics"),
    ("Football is a popular game", "sports"),
    ("Elections are coming soon", "politics"),
    ("The team celebrated their victory", "sports"),
    ("The government announced new policies", "politics"),
    ("Basketball is played worldwide", "sports"),
    ("The opposition criticized the budget", "politics"),
    ("Tennis tournaments are exciting", "sports"),
    ("The president addressed the nation", "politics"),
    ("Hockey is a winter sport", "sports"),
    ("Political debates are often heated", "politics"),
    ("Athletes train hard for competitions", "sports"),
    ("New laws were passed in parliament", "politics"),
    ("Cricket fans are passionate", "sports"),
    ("The senator spoke about healthcare", "politics"),
    ("Rugby is gaining popularity", "sports"),
    ("The mayor discussed city plans", "politics"),
    ("Volleyball is a team sport", "sports"),
    ("The cabinet met to discuss reforms", "politics"),
    ("Badminton is played in many countries", "sports"),
    ("The election results were announced", "politics"),
    ("Cycling is a great way to stay fit", "sports"),
    ("The parliament debated the new bill", "politics")
]

# Step 1: Count words in each class
from collections import defaultdict

word_freq = {
    "sports": defaultdict(int),
    "politics": defaultdict(int)
}
class_count = {"sports": 0, "politics": 0}

for sentence, label in train_data:
    words = sentence.lower().split()
    class_count[label] += 1
    for word in words:
        word_freq[label][word] += 1


# Step 2: Predict
def predict(sentence):
    words = sentence.lower().split()
    scores = {}

    for label in ["sports", "politics"]:
        # Start with the prior probability
        score = class_count[label] / sum(class_count.values())
        for word in words:
            # Likelihood (add-one smoothing)
            word_prob = (word_freq[label][word] + 1) / (sum(word_freq[label].values()) + len(word_freq[label]))
            score *= word_prob
        scores[label] = score
    print("Scores:", scores)  # Debugging line to see scores
    # Return the class with the highest score
    return max(scores, key=scores.get)

# Test
while True:
    try:
        print(predict(input("Enter your thoughts : ")))  # sports or politics? 🤔
    except KeyboardInterrupt:
        break
