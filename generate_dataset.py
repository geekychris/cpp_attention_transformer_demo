#!/usr/bin/env python3
"""
Generate training data for the C++ transformer model.
This script creates both synthetic data and can download real datasets.
"""

import requests
import os
import random
import sys

def download_tiny_shakespeare():
    """Download the tiny shakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    try:
        print("Downloading tiny shakespeare dataset...")
        response = requests.get(url)
        response.raise_for_status()
        
        with open("shakespeare.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Downloaded shakespeare.txt ({len(response.text)} characters)")
        return True
    except Exception as e:
        print(f"Failed to download shakespeare dataset: {e}")
        return False

def generate_synthetic_stories():
    """Generate synthetic story data for training."""
    
    # Simple story templates
    characters = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
    locations = ["forest", "castle", "village", "mountain", "river", "cave", "garden", "library"]
    objects = ["sword", "book", "key", "treasure", "map", "crown", "ring", "crystal"]
    actions = ["found", "lost", "searched for", "discovered", "protected", "stole", "gave", "hid"]
    adjectives = ["ancient", "magical", "mysterious", "golden", "silver", "powerful", "hidden", "sacred"]
    
    stories = []
    
    # Generate simple stories
    for _ in range(1000):
        char1 = random.choice(characters)
        char2 = random.choice([c for c in characters if c != char1])
        location = random.choice(locations)
        obj = random.choice(objects)
        action = random.choice(actions)
        adj = random.choice(adjectives)
        
        # Simple story templates
        templates = [
            f"Once upon a time, {char1} lived in a {location}. One day, {char1} {action} a {adj} {obj}. {char2} helped {char1} find it.",
            f"In the {adj} {location}, {char1} and {char2} searched for the {obj}. They {action} it after many adventures.",
            f"{char1} was a brave hero who {action} the {adj} {obj} from the {location}. {char2} was very grateful.",
            f"The {adj} {obj} was hidden in the {location}. {char1} and {char2} worked together to find it.",
            f"{char2} told {char1} about the legendary {obj} in the {location}. Together they {action} it."
        ]
        
        story = random.choice(templates)
        stories.append(story)
    
    return stories

def generate_simple_sentences():
    """Generate simple sentences for basic training."""
    
    subjects = ["the cat", "the dog", "the bird", "the fish", "the mouse", "the rabbit"]
    verbs = ["runs", "jumps", "sleeps", "eats", "plays", "walks", "sits", "flies"]
    objects = ["in the garden", "on the table", "under the tree", "near the house", "by the river", "in the park"]
    
    sentences = []
    
    for _ in range(2000):
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        
        sentence = f"{subject} {verb} {obj}."
        sentences.append(sentence)
    
    return sentences

def create_training_data():
    """Create comprehensive training data."""
    
    all_text = []
    
    # Try to download shakespeare
    if download_tiny_shakespeare():
        with open("shakespeare.txt", "r", encoding="utf-8") as f:
            shakespeare_text = f.read()
        
        # Clean up shakespeare text a bit
        lines = shakespeare_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.isupper() and len(line) > 10:  # Skip headers and very short lines
                cleaned_lines.append(line)
        
        all_text.extend(cleaned_lines[:5000])  # Use first 5000 good lines
    
    # Add synthetic stories
    stories = generate_synthetic_stories()
    all_text.extend(stories)
    
    # Add simple sentences
    sentences = generate_simple_sentences()
    all_text.extend(sentences)
    
    # Shuffle all text
    random.shuffle(all_text)
    
    # Split into train/validation
    split_idx = int(0.9 * len(all_text))
    train_text = all_text[:split_idx]
    val_text = all_text[split_idx:]
    
    # Write training data
    with open("train_data.txt", "w", encoding="utf-8") as f:
        for line in train_text:
            f.write(line + "\n")
    
    with open("val_data.txt", "w", encoding="utf-8") as f:
        for line in val_text:
            f.write(line + "\n")
    
    print(f"Created training dataset:")
    print(f"  - train_data.txt: {len(train_text)} lines")
    print(f"  - val_data.txt: {len(val_text)} lines")
    
    # Create a small test set for quick inference testing
    test_prompts = [
        "Once upon a time",
        "The brave knight",
        "In the forest",
        "Alice found a",
        "The magical sword",
        "the cat runs",
        "the dog sleeps"
    ]
    
    with open("test_prompts.txt", "w", encoding="utf-8") as f:
        for prompt in test_prompts:
            f.write(prompt + "\n")
    
    print(f"  - test_prompts.txt: {len(test_prompts)} test prompts")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    create_training_data()
    print("Dataset generation complete!")