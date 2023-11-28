import pandas as pd

# Create a DataFrame for the given dataset
data = {
    'Day': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14'],
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Calculate prior probabilities P(Yes) and P(No)
total_instances = df.shape[0]
num_yes = df[df['Play'] == 'Yes'].shape[0]
num_no = df[df['Play'] == 'No'].shape[0]
p_yes = num_yes / total_instances
p_no = num_no / total_instances

# Define the new data point
new_data = {'Outlook': 'Rain', 'Temp': 'Hot', 'Humidity': 'Normal', 'Wind': 'Strong'}

# Calculate the conditional probabilities for each attribute given the class
conditional_probs = {}
for col in df.columns[1:-1]:  # Exclude 'Day' and 'Play' columns
    for label in df[col].unique():
        for play in df['Play'].unique():
            num_with_label_and_play = df[(df[col] == label) & (df['Play'] == play)].shape[0]
            num_with_play = df[df['Play'] == play].shape[0]
            prob_key = f'P({col}={label}|Play={play})'
            conditional_probs[prob_key] = num_with_label_and_play / num_with_play

# Calculate the posterior probabilities
posterior_probs = {}
for play in df['Play'].unique():
    posterior_prob = p_yes if play == 'Yes' else p_no
    for col, label in new_data.items():
        prob_key = f'P({col}={label}|Play={play})'
        posterior_prob *= conditional_probs[prob_key]
    posterior_probs[play] = posterior_prob

# Classify the new data point
predicted_class = max(posterior_probs, key=posterior_probs.get)
print(f"The predicted class for the new data point is: {predicted_class}")
