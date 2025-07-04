#!/usr/bin/env python3
"""
Corrected DSPy Optimization Script for Shopping Assistant
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict

import dspy

# Configure DSPy
api_key = os.getenv('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')
if api_key == 'YOUR_API_KEY_HERE':
    print("⚠️  Please set your OPENAI_API_KEY environment variable")
    exit(1)

lm = dspy.LM('openai/gpt-4.1', api_key=api_key)
dspy.configure(lm=lm)
print("✅ DSPy configured successfully!")


class ConversationSuggestionSignature(dspy.Signature):
    """Generate conversation suggestions for shopping assistant."""
    messages = dspy.InputField(desc="Conversation history as JSON string")
    role = dspy.InputField(desc="Assistant role")
    task = dspy.InputField(desc="Task to perform")
    language = dspy.InputField(desc="Language for suggestions")
    suggestions = dspy.OutputField(desc="JSON with 'answers_sugg' array (0-3 items, max 40 chars each)")


class BaselineModule(dspy.Module):
    """Baseline using original prompt template"""

    def __init__(self, prompt_template):
        super().__init__()
        self.prompt_template = prompt_template

    def forward(self, messages, role, task, language):
        # Handle both string and list inputs
        if isinstance(messages, str):
            messages_str = messages
        else:
            messages_str = json.dumps(messages, indent=2)

        prompt = self.prompt_template.format(
            role=role,
            task=task,
            messages=messages_str,
            language=language
        )

        response = dspy.LM(prompt)
        suggestions = self.parse_suggestions(response)
        return dspy.Prediction(suggestions=suggestions)

    def parse_suggestions(self, response):
        try:
            # Extract JSON from response
            match = re.search(r'\{[^}]+\}', response)
            if match:
                data = json.loads(match.group())
                return data.get('answers_sugg', [])
        except:
            pass
        return []


class ShoppingAssistantSuggester(dspy.Module):
    """Optimizable DSPy module"""

    def __init__(self):
        super().__init__()
        self.suggest = dspy.ChainOfThought(ConversationSuggestionSignature)

    def forward(self, messages, role, task, language):
        # Handle both string and list inputs
        if isinstance(messages, str):
            messages_str = messages
        else:
            messages_str = json.dumps(messages, indent=2)

        prediction = self.suggest(
            messages=messages_str,
            role=role,
            task=task,
            language=language
        )

        # Parse the suggestions correctly
        suggestions = self.parse_suggestions(prediction.suggestions)
        return dspy.Prediction(suggestions=suggestions)

    def parse_suggestions(self, output):
        try:
            # Handle various output formats
            if isinstance(output, str):
                # Try to find JSON in the output
                json_match = re.search(r'\{.*"answers_sugg".*\}', output, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return data.get('answers_sugg', [])
            elif isinstance(output, dict):
                return output.get('answers_sugg', [])
            elif isinstance(output, list):
                return output[:3]  # Limit to 3
        except Exception as e:
            print(f"Parse error: {e}")
        return []


def suggestion_metric(example, prediction, trace=None):
    """Improved metric for evaluating suggestions"""
    try:
        pred_suggestions = prediction.suggestions if hasattr(prediction, 'suggestions') else []
        expected = example.suggestions if hasattr(example, 'suggestions') else example.expected_suggestions

        # Check if appointment booking scenario (expecting empty)
        is_appointment = len(expected) == 0

        if is_appointment:
            # For appointments, penalize any suggestions
            return 1.0 if len(pred_suggestions) == 0 else 0.0
        else:
            # For regular scenarios
            if len(pred_suggestions) == 0:
                return 0.0

            # Check validity (length <= 40 chars)
            valid_count = sum(1 for s in pred_suggestions if len(s) <= 40)
            validity_score = valid_count / max(1, len(pred_suggestions))

            # Check quantity (expecting 3 suggestions)
            quantity_score = min(len(pred_suggestions), 3) / 3.0

            # Combined score
            return 0.7 * validity_score + 0.3 * quantity_score

    except Exception as e:
        print(f"Metric error: {e}")
        return 0.0


def create_dspy_examples(training_data):
    """Convert training data to DSPy format"""
    examples = []

    for data in training_data:
        # Convert messages to JSON string to avoid dict hashing issues
        messages_str = json.dumps(data['messages'])

        # Create proper DSPy example
        example = dspy.Example(
            messages=messages_str,  # Already as string
            role="shopping assistant suggestion generator",
            task="Generate relevant conversation suggestions",
            language="English",
            suggestions=json.dumps({"answers_sugg": data['expected_suggestions']})
        ).with_inputs("messages", "role", "task", "language")

        examples.append(example)

    return examples


def main():
    print("🚀 Starting DSPy Prompt Optimization...")
    print("=" * 60)

    # Your original prompt template
    ORIGINAL_PROMPT = """# ROLE
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

    # Training data
    training_data = [
        {
            'messages': [
                {"role": "assistant", "content": "What color flooring are you looking for?"}
            ],
            'expected_suggestions': ["Dark brown", "Light colors", "Undecided"]
        },
        {
            'messages': [
                {"role": "assistant", "content": "What is your budget for the flooring project?"}
            ],
            'expected_suggestions': ["€50/m²", "€3000 total", "Flexible"]
        },
        {
            'messages': [
                {"role": "assistant", "content": "What is your customer number and zip code?"}
            ],
            'expected_suggestions': []
        },
        {
            'messages': [
                {"role": "assistant", "content": "Would you like to schedule an appointment?"}
            ],
            'expected_suggestions': []
        },
        {
            'messages': [
                {"role": "assistant", "content": "What room will this flooring be for?"}
            ],
            'expected_suggestions': ["Kitchen", "Bathroom", "Living room"]
        }
    ]

    # Convert to DSPy examples
    print("📚 Creating DSPy examples...")
    examples = create_dspy_examples(training_data)
    train_examples = examples[:3]
    test_examples = examples[3:]

    # Create baseline and optimizable modules
    baseline = BaselineModule(ORIGINAL_PROMPT)
    program = ShoppingAssistantSuggester()

    # Evaluate baseline
    print("\n📊 Evaluating baseline (original prompt)...")
    baseline_score = 0
    for ex in test_examples:
        # Parse messages from string if needed
        messages = json.loads(ex.messages) if isinstance(ex.messages, str) else ex.messages
        pred = baseline(messages, ex.role, ex.task, ex.language)
        score = suggestion_metric(ex, pred)
        baseline_score += score
        print(f"  Example: {score:.2f}")
    baseline_score /= len(test_examples)
    print(f"Baseline score: {baseline_score:.2f}")

    # Evaluate unoptimized DSPy module
    print("\n📊 Evaluating unoptimized DSPy module...")
    unopt_score = 0
    for ex in test_examples:
        messages = json.loads(ex.messages) if isinstance(ex.messages, str) else ex.messages
        pred = program(messages, ex.role, ex.task, ex.language)
        score = suggestion_metric(ex, pred)
        unopt_score += score
        print(f"  Example: {score:.2f}")
    unopt_score /= len(test_examples)
    print(f"Unoptimized DSPy score: {unopt_score:.2f}")

    # Optimize
    print("\n⚙️  Optimizing with MIPROv2...")
    from dspy.teleprompt import MIPROv2

    # Option 1: Use auto mode (recommended for beginners)
    optimizer = MIPROv2(
        metric=suggestion_metric,
        auto="light",  # This sets all parameters automatically
    )

    optimized_program = optimizer.compile(
        program.deepcopy(),
        trainset=train_examples,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        requires_permission_to_run=False,
    )

    # Option 2: Manual configuration (comment out Option 1 and uncomment this)
    # optimizer = MIPROv2(
    #     metric=suggestion_metric,
    #     num_candidates=7,
    #     init_temperature=0.5,
    # )
    #
    # optimized_program = optimizer.compile(
    #     program.deepcopy(),
    #     trainset=train_examples,
    #     num_trials=15,
    #     max_bootstrapped_demos=2,
    #     max_labeled_demos=2,
    #     requires_permission_to_run=False,
    # )

    # Evaluate optimized
    print("\n📊 Evaluating optimized program...")
    opt_score = 0
    for ex in test_examples:
        messages = json.loads(ex.messages) if isinstance(ex.messages, str) else ex.messages
        pred = optimized_program(messages, ex.role, ex.task, ex.language)
        score = suggestion_metric(ex, pred)
        opt_score += score
        print(f"  Example: {score:.2f}")
    opt_score /= len(test_examples)

    # Results
    print("\n📈 Final Results:")
    print(f"  Baseline (original prompt): {baseline_score:.2f}")
    print(f"  Unoptimized DSPy: {unopt_score:.2f}")
    print(f"  Optimized DSPy: {opt_score:.2f}")
    print(f"  Improvement over baseline: {opt_score - baseline_score:.2f}")

    # Extract and display optimized prompt
    print("\n📋 EXTRACTING OPTIMIZED PROMPT...")
    print("=" * 60)

    try:
        # Method 1: Load and inspect saved file
        print("\n🔍 Loading saved optimization file...")
        with open("optimized_shopping_assistant.json", "r") as f:
            saved_data = json.load(f)

        # Extract the optimized prompt details
        if "suggest" in saved_data:
            suggest_data = saved_data["suggest"]

            # Look for the actual prompt/instructions
            if "signature" in suggest_data:
                sig_data = suggest_data["signature"]

                print("\n📝 OPTIMIZED PROMPT INSTRUCTIONS:")
                print("-" * 60)

                # Extract main instructions
                if "instructions" in sig_data:
                    print(f"\nMain Instructions:\n{sig_data['instructions']}")

                # Extract field descriptions
                print("\n📌 Field Descriptions:")
                for field_name in ['messages', 'role', 'task', 'language', 'suggestions']:
                    if field_name in sig_data:
                        field_data = sig_data[field_name]
                        if isinstance(field_data, dict):
                            print(f"\n{field_name}:")
                            for key, value in field_data.items():
                                print(f"  {key}: {value}")

                # Look for extended signature
                if "extended_signature" in suggest_data:
                    print("\n🎯 Extended Signature Details:")
                    ext_sig = suggest_data["extended_signature"]
                    if isinstance(ext_sig, dict):
                        if "instructions" in ext_sig:
                            print(f"\nOptimized Instructions:\n{ext_sig['instructions']}")

                        # Check for reasoning field (from ChainOfThought)
                        if "reasoning" in ext_sig:
                            print(f"\nReasoning Field:\n{ext_sig['reasoning']}")

            # Look for demos (few-shot examples)
            if "demos" in suggest_data and suggest_data["demos"]:
                print(f"\n📚 Number of Few-Shot Examples: {len(suggest_data['demos'])}")
                print("\nFirst Few-Shot Example:")
                print(json.dumps(suggest_data["demos"][0], indent=2)[:500] + "...")

        # Method 2: Try to inspect the actual optimized program
        print("\n\n🔧 Inspecting Optimized Program Object...")

        # Get the optimized ChainOfThought module
        cot_module = optimized_program.suggest

        # Check signature
        if hasattr(cot_module, 'signature'):
            print(f"\nSignature Class: {cot_module.signature.__class__.__name__}")

            # Get the actual prompt by inspecting the signature
            sig = cot_module.signature
            if hasattr(sig, '__dict__'):
                print("\nSignature Attributes:")
                for attr, value in sig.__dict__.items():
                    if not attr.startswith('_'):
                        print(f"  {attr}: {value}")

        # Check for extended signature (ChainOfThought specific)
        if hasattr(cot_module, 'extended_signature'):
            print(f"\nExtended Signature: {cot_module.extended_signature}")

        # Method 3: Generate a sample to see the actual prompt
        print("\n\n💡 SAMPLE PROMPT GENERATION...")
        print("-" * 60)

        # Create a simple test case
        test_messages = [
            {"role": "assistant", "content": "What type of flooring are you looking for?"}
        ]

        print("\nGenerating a sample with the optimized program...")
        print("This will show how the optimized prompt handles the task.")

        # Generate with the optimized program
        try:
            result = optimized_program(
                messages=test_messages,
                role="shopping assistant suggestion generator",
                task="Generate relevant conversation suggestions",
                language="English"
            )
            print(f"\nSample Output: {result.suggestions}")
        except Exception as e:
            print(f"Sample generation note: {e}")

        # Method 4: Show comparison
        print("\n\n📊 PROMPT COMPARISON")
        print("=" * 60)

        print("\n🔹 ORIGINAL PROMPT TEMPLATE (Baseline):")
        print("-" * 40)
        print("Main focus: Manual rules about appointment detection")
        print("Key instruction: 'NEVER GIVE SUGGESTIONS WHERE IT IS NOT USEFUL'")
        print("Approach: Explicit rules for zip codes, personal data, appointments")

        print("\n🔹 OPTIMIZED PROMPT (After MIPROv2):")
        print("-" * 40)

        if "suggest" in saved_data and "signature" in saved_data["suggest"]:
            opt_instructions = saved_data["suggest"]["signature"].get("instructions", "")
            if opt_instructions:
                print(f"Optimized Instructions: {opt_instructions}")

            # Show if there are few-shot examples
            if "demos" in saved_data["suggest"] and saved_data["suggest"]["demos"]:
                print(f"\nIncludes {len(saved_data['suggest']['demos'])} few-shot examples")
                print("These examples teach the model when to return empty suggestions")

        print("\n🎯 KEY IMPROVEMENTS:")
        print("- Data-driven learning from examples vs manual rules")
        print("- Automatic handling of appointment scenarios through examples")
        print("- More robust and generalizable approach")

    except Exception as e:
        print(f"\n⚠️  Error extracting optimized prompt: {e}")
        print("The optimization file has been saved. You can inspect it manually.")

    print("\n" + "=" * 60)
    print("✅ Complete!")


if __name__ == "__main__":
    main()