import re

knowledge = {}

def teach(sentence):
    # Basic pattern: "X is Y" → store {X: Y}
    match = re.match(r"(.*) is (.*)", sentence, re.IGNORECASE)
    if match:
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        knowledge[key] = value
        print("Learned:", key, "->", value)
    else:
        print("Sorry, I only understand sentences like 'X is Y'.")

def ask(question):
    # Normalize question
    q = question.strip().lower().replace("?", "")
    print("You asked:", q)
    # Match: "what is X"
    match = re.match(r"what is (.*)", q)
    if match:
        key = match.group(1).strip()
        if key in knowledge:
            print(knowledge[key])
        else:
            print("I don't know.")
    else:
        # Try: "who is", "where is", etc.
        for k in knowledge:
            if k in q:
                print(knowledge[k])
                return
        print("I don't understand the question.")

def find_prompt_type(prompt):
    # Check if the prompt is a teaching or asking prompt
    if re.match(r"what is .*", prompt, re.IGNORECASE):
        ask(prompt)
    elif re.match(r".* is .*", prompt, re.IGNORECASE):
        teach(prompt)
    else:
        return print("Can only teach or ask questions like 'X is Y' or 'What is X'.")
    

# Chat loop
while True:
    try:
        find_prompt_type(input("You: ").strip())
    except KeyboardInterrupt:
        print("\nExiting chat. Goodbye!")
        break
