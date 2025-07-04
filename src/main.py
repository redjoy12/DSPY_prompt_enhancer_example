#!/usr/bin/env python3
"""
Runnable DSPy Optimization Script for Shopping Assistant
Run this script after configuring your OpenAI API key
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict

import dspy

# Configure DSPy with your API key
# Make sure to set your OPENAI_API_KEY environment variable
# or replace 'YOUR_API_KEY_HERE' with your actual key
api_key = os.getenv('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')
if api_key == 'YOUR_API_KEY_HERE':
    print("⚠️  Please set your OPENAI_API_KEY environment variable or update the script")
    print("   export OPENAI_API_KEY='your-key-here'")
    exit(1)

lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
dspy.configure(lm=lm)
print("✅ DSPy configured successfully!")


@dataclass
class ConversationExample:
    """Training example for the conversation suggestion system"""
    messages: List[Dict[str, str]]
    expected_suggestions: List[str]
    language: str = "English"
    role: str = "shopping assistant suggestion generator"
    task: str = "Generate relevant conversation suggestions"


# Your original prompt template (BASELINE)
ORIGINAL_PROMPT_TEMPLATE = """# ROLE
You are a {role}. Your goal is to help a customer continue their conversation effectively with shopping assistant.

# TASK
{task}

# INPUT
You will receive the conversation history as a list of messages. Each message has a 'role' (user or assistant) and 'content' (the text of the message).

{messages}

# INSTRUCTIONS
- Analyze History: Review the entire messages list to understand the conversation context, customer preferences already stated, and the flow of the discussion.
- Focus on Last Message: Pay primary attention to the very last message in the list (from the assistant). Your suggestions should directly address or respond to this specific message.
- Prioritize Direct Answers: If the assistant's last message is a direct question, the suggestions should primarily be direct answers.
- Maximize Conciseness (Keyword Preference): Strive to make suggestions as concise as possible. Prefer keywords or very short phrases (e.g., "Living room", "Dark brown", "€50/m²", "Yes", "More details please") over full sentences, especially when providing direct answers, whenever this is natural and contextually appropriate.
- Clarity: Despite brevity, each suggestion must be clear.
- Character Limit: Each suggestion must have a maximum length of 40 characters.
- Relevance: Suggestions must be highly relevant to the assistant's last message and consistent with the overall conversation history.
- Avoid Redundancy: Do not suggest questions or statements about information the user has already clearly provided earlier in the chat (e.g., if color preference is known, don't suggest colors unless the assistant specifically asks again or introduces a new context for it).
- Currency: If prices are discussed, use Euros (€). Consider realistic price ranges for the item in context when suggesting budget answers.
- Language: Generate the suggestions in {language}.
- NEVER GIVE SUGGESTIONS WHERE IT IS NOT USEFUL: Evaluate if it is likely that one of the suggestions would be selected by the customer. For instance, if the customer is asked for a zip code, personal data, or any other specifics, this is very unlikely because of the high number of possibilities. In this case, ALWAYS generate an empty list of suggestions.

# OUTPUT FORMAT
Provide the output as a JSON object with "answers_sugg" containing exactly 3 (IF USEFUL suggestions can be made) OR 0 (IF NO USEFUL suggestions can be made) suggestion strings. Do not include any extra explanations or introductory text.

Response with JSON only: {{"answers_sugg": ["suggestion1", "suggestion2", "suggestion3"]}} or {{"answers_sugg": []}}
"""


class BaselineModule(dspy.Module):
    """Baseline module using your original prompt template"""

    def __init__(self):
        super().__init__()
        self.lm = dspy.settings.lm

    def forward(self, messages: List[Dict[str, str]], role: str, task: str, language: str):
        # Format your original prompt
        messages_str = json.dumps(messages, indent=2)
        formatted_prompt = ORIGINAL_PROMPT_TEMPLATE.format(
            role=role,
            task=task,
            messages=messages_str,
            language=language
        )

        # Get response from LM
        response = self.lm(formatted_prompt)
        suggestions = self.parse_suggestions(response)

        return dspy.Prediction(suggestions=suggestions)

    def parse_suggestions(self, response_text: str) -> List[str]:
        """Parse suggestions from the original prompt's output"""
        try:
            # Try to parse as JSON first
            if '{' in response_text:
                # Extract JSON part
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_part = response_text[start:end]
                parsed = json.loads(json_part)
                if 'answers_sugg' in parsed:
                    return parsed['answers_sugg']

            # Fallback: extract suggestions from text
            return []
        except Exception as e:
            print(f"Error parsing baseline response: {e}")
            return []


class ConversationSuggestionSignature(dspy.Signature):
    """
    Generate conversation suggestions for a shopping assistant based on conversation history.
    Focus on the last assistant message and provide relevant, concise suggestions.
    Return empty suggestions for appointment booking scenarios.
    """
    messages = dspy.InputField(desc="List of conversation messages with role and content")
    role = dspy.InputField(desc="The role of the assistant")
    task = dspy.InputField(desc="The specific task to perform")
    language = dspy.InputField(desc="Language for suggestions")
    suggestions = dspy.OutputField(
        desc="JSON object with 'answers_sugg' containing 0-3 suggestions, each max 40 characters. Return empty array for appointment booking.")


class ShoppingAssistantSuggester(dspy.Module):
    """DSPy Module for generating shopping assistant conversation suggestions"""

    def __init__(self):
        super().__init__()
        self.suggest = dspy.ChainOfThought(ConversationSuggestionSignature)

    def forward(self, messages: List[Dict[str, str]], role: str, task: str, language: str):
        # Convert messages to string format for processing
        messages_str = json.dumps(messages, indent=2)

        # Generate suggestions
        prediction = self.suggest(
            messages=messages_str,
            role=role,
            task=task,
            language=language
        )

        # Parse suggestions from the prediction
        suggestions = self.parse_suggestions(prediction.suggestions)

        return dspy.Prediction(suggestions=suggestions)

    def parse_suggestions(self, suggestions_text: str) -> List[str]:
        """Parse suggestions from model output"""
        try:
            # Try to parse as JSON first
            if suggestions_text.strip().startswith('{'):
                parsed = json.loads(suggestions_text)
                if 'answers_sugg' in parsed:
                    return parsed['answers_sugg']
            elif suggestions_text.strip().startswith('['):
                return json.loads(suggestions_text)

            # Fallback: extract suggestions from text
            lines = suggestions_text.strip().split('\n')
            suggestions = []
            for line in lines:
                # Remove numbering, quotes, and clean up
                clean_line = re.sub(r'^\d+\.?\s*', '', line).strip()
                clean_line = clean_line.strip('"\'')
                if clean_line and len(clean_line) <= 40:
                    suggestions.append(clean_line)

            return suggestions[:3]  # Max 3 suggestions
        except Exception as e:
            print(f"Error parsing suggestions: {e}")
            return []


def create_training_data() -> List[ConversationExample]:
    """Create training data including appointment booking scenarios"""

    training_examples = [
        # Original flooring examples
        ConversationExample(
            messages=[
                {"role": "assistant",
                 "content": "Hello! I'd be happy to help you find the perfect flooring. Let me gather some information about different types of flooring for you. One moment, please."},
                {"role": "assistant",
                 "content": "I have gathered some general information...\nTypes of Flooring:\n - Laminate...\n - Parquet...\n - Vinyl...\n - Solid Wood Planks...\n - Cork...\nFactors to Consider:\n - Budget...\n - Durability...\n - Installation...\n - Maintenance...\n - Aesthetics...\n - Special Needs...\nCould you tell me what color you are looking for in the flooring?"}
            ],
            expected_suggestions=["Dark brown", "Light colors", "Undecided on color"]
        ),

        ConversationExample(
            messages=[
                {"role": "assistant",
                 "content": "Hello! I'd be happy to help you find the perfect flooring. Let me gather some information about different types of flooring for you. One moment, please."},
                {"role": "assistant",
                 "content": "I have gathered some general information...\nTypes of Flooring:\n - Laminate...\n - Parquet...\n - Vinyl...\n - Solid Wood Planks...\n - Cork...\nFactors to Consider:\n - Budget...\n - Durability...\n - Installation...\n - Maintenance...\n - Aesthetics...\n - Special Needs...\nWhat is your budget for the flooring project?"}
            ],
            expected_suggestions=["€50/m²", "€3000 total", "Flexible budget"]
        ),

        ConversationExample(
            messages=[
                {"role": "assistant",
                 "content": "Hello! I'd be happy to help you find the perfect flooring. Let me gather some information about different types of flooring for you. One moment, please."},
                {"role": "assistant",
                 "content": "I have gathered some general information...\nTypes of Flooring:\n - Laminate...\n - Parquet...\n - Vinyl...\n - Solid Wood Planks...\n - Cork...\nFactors to Consider:\n - Budget...\n - Durability...\n - Installation...\n - Maintenance...\n - Aesthetics...\n - Special Needs...\nWhat is your customer number and zip code, so I can give you tailored recommendations?"}
            ],
            expected_suggestions=[]
        ),

        # NEW: Appointment booking scenarios
        ConversationExample(
            messages=[
                {"role": "user", "content": "I'm interested in this flooring option."},
                {"role": "assistant",
                 "content": "Great choice! This laminate flooring is very popular. Would you like to schedule an appointment for a consultation?"}
            ],
            expected_suggestions=[]
        ),

        ConversationExample(
            messages=[
                {"role": "user", "content": "Can someone come to my house to measure?"},
                {"role": "assistant",
                 "content": "Absolutely! We offer free in-home measurements. When would be a good time to book an appointment for the measurement?"}
            ],
            expected_suggestions=[]
        ),

        ConversationExample(
            messages=[
                {"role": "user", "content": "I need someone to install this flooring."},
                {"role": "assistant",
                 "content": "We have professional installers available. Let me help you book an installation appointment. What's your preferred date?"}
            ],
            expected_suggestions=[]
        ),

        ConversationExample(
            messages=[
                {"role": "user", "content": "When can I schedule a visit?"},
                {"role": "assistant",
                 "content": "I can help you schedule a visit from our flooring experts. Please let me know your availability and we'll find a suitable appointment time."}
            ],
            expected_suggestions=[]
        ),

        ConversationExample(
            messages=[
                {"role": "user", "content": "Can we set up a meeting?"},
                {"role": "assistant",
                 "content": "Of course! I'd be happy to arrange a meeting with our flooring specialists. What would work best for your schedule?"}
            ],
            expected_suggestions=[]
        ),

        # Regular shopping scenarios (should have suggestions)
        ConversationExample(
            messages=[
                {"role": "user", "content": "I'm looking for durable flooring for my kitchen."},
                {"role": "assistant",
                 "content": "For kitchen flooring, durability and water resistance are key. What style are you looking for?"}
            ],
            expected_suggestions=["Modern", "Traditional", "Rustic"]
        ),

        ConversationExample(
            messages=[
                {"role": "user", "content": "Tell me about vinyl flooring."},
                {"role": "assistant",
                 "content": "Vinyl flooring is waterproof, durable, and comes in many styles. What room will this be for?"}
            ],
            expected_suggestions=["Kitchen", "Bathroom", "Living room"]
        ),

        ConversationExample(
            messages=[
                {"role": "user", "content": "How much does laminate cost?"},
                {"role": "assistant",
                 "content": "Laminate flooring typically ranges from €25-€80 per m². How large is the area you need to cover?"}
            ],
            expected_suggestions=["Small room", "Whole house", "Don't know yet"]
        )
    ]

    return training_examples


def appointment_booking_metric(example: ConversationExample, prediction, trace=None) -> float:
    """
    Custom metric that evaluates whether the system correctly handles appointment booking scenarios.
    Returns 1.0 for correct behavior, 0.0 for incorrect behavior.
    """
    try:
        predicted_suggestions = prediction.suggestions if hasattr(prediction, 'suggestions') else []
        expected_suggestions = example.expected_suggestions

        # Check if this is an appointment booking scenario
        is_appointment_scenario = len(expected_suggestions) == 0

        if is_appointment_scenario:
            # For appointment booking, we want empty suggestions
            score = 1.0 if len(predicted_suggestions) == 0 else 0.0
            print(f"   Appointment scenario: Expected=[], Got={predicted_suggestions}, Score={score}")
            return score
        else:
            # For regular scenarios, we want non-empty relevant suggestions
            if len(predicted_suggestions) == 0:
                print(f"   Regular scenario: Expected={expected_suggestions}, Got=[], Score=0.0")
                return 0.0

            # Basic relevance and length check
            valid_suggestions = [s for s in predicted_suggestions if len(s) <= 40]
            score = len(valid_suggestions) / max(1, len(predicted_suggestions))
            print(f"   Regular scenario: Expected={expected_suggestions}, Got={predicted_suggestions}, Score={score}")
            return score

    except Exception as e:
        print(f"   Error in metric evaluation: {e}")
        return 0.0


def evaluate_program(program, test_examples: List[ConversationExample]) -> float:
    """Evaluate the program on test examples"""
    total_score = 0.0
    print("\nEvaluating program...")

    for i, example in enumerate(test_examples):
        try:
            print(f"\nExample {i + 1}: {example.messages[-1]['content'][:50]}...")
            prediction = program(
                messages=example.messages,
                role=example.role,
                task=example.task,
                language=example.language
            )
            score = appointment_booking_metric(example, prediction)
            total_score += score
        except Exception as e:
            print(f"   Error evaluating example {i + 1}: {e}")

    return total_score / len(test_examples) if test_examples else 0.0


def main():
    """Main function to optimize the prompt using DSPy MIPROv2"""
    print("🚀 Starting DSPy Prompt Optimization...")
    print("=" * 60)

    # Create training data
    print("📚 Creating training data...")
    training_data = create_training_data()

    # Split data into train and test
    train_examples = training_data[:8]  # Use first 8 for training
    test_examples = training_data[8:]  # Use remaining for testing

    print(f"   Training examples: {len(train_examples)}")
    print(f"   Test examples: {len(test_examples)}")

    # Create the initial program
    print("\n🔧 Creating initial program...")
    program = ShoppingAssistantSuggester()

    # Evaluate baseline performance
    print("\n📊 Evaluating baseline performance...")
    baseline_score = evaluate_program(program, test_examples)
    print(f"\n📈 Baseline score: {baseline_score:.2f}")

    # Convert training examples to DSPy format
    print("\n🔄 Converting training data to DSPy format...")
    dspy_trainset = []
    for example in train_examples:
        dspy_example = dspy.Example(
            messages=example.messages,
            role=example.role,
            task=example.task,
            language=example.language,
            suggestions=example.expected_suggestions
        ).with_inputs("messages", "role", "task", "language")
        dspy_trainset.append(dspy_example)

    # Import and configure optimizer
    print("\n⚙️  Setting up MIPROv2 optimizer...")
    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(
        metric=appointment_booking_metric,
        auto=None,
        num_candidates=5,
        init_temperature=0.5,
    )

    # Optimize the program
    print("\n🎯 Optimizing program with MIPROv2...")
    print("   This may take a few minutes...")

    try:
        optimized_program = optimizer.compile(
            program,
            trainset=dspy_trainset,
            num_trials=10,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            requires_permission_to_run=False,
            minibatch_size=6
        )

        # Evaluate optimized performance
        print("\n📊 Evaluating optimized performance...")
        optimized_score = evaluate_program(optimized_program, test_examples)
        print(f"\n📈 Results:")
        print(f"   Baseline score: {baseline_score:.2f}")
        print(f"   Optimized score: {optimized_score:.2f}")
        print(f"   Improvement: {optimized_score - baseline_score:.2f}")

        # Save the optimized program
        print("\n💾 Saving optimized program...")
        optimized_program.save("optimized_shopping_assistant.json")
        print("   Saved to: optimized_shopping_assistant.json")

        # Extract and display the optimized prompt
        print("\n📋 Extracting optimized prompt...")
        try:
            # Get the optimized prompt from the first module
            suggest_module = optimized_program.suggest
            if hasattr(suggest_module, 'signature'):
                optimized_prompt_template = str(suggest_module.signature)
                print("\n🎉 OPTIMIZED PROMPT EXTRACTED:")
                print("=" * 60)
                print(optimized_prompt_template)
            else:
                print("   Could not extract prompt template automatically.")
                print("   Check the saved file: optimized_shopping_assistant.json")
        except Exception as e:
            print(f"   Error extracting prompt: {e}")
            print("   Check the saved file: optimized_shopping_assistant.json")

        print("\n✅ Optimization complete!")

    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        print("   Check your API key and try again.")


if __name__ == "__main__":
    main()
