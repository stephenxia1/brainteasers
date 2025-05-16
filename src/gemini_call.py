import re
from google import genai
from google.genai import types
from anthropic import AnthropicVertex

def parse_blackjack_response(response):
    # Improved patterns for stability
    move_pattern = r'\[best_move\]:?\s*(-?\d+)'  # Matches [best_move] or [best_move]: followed by a number
    rationale_pattern = r'\[(rationale|ratioale)\]:?\s*((?:(?!\[best_move\]).)*)'  # Matches all variations of [rationale] or [ratioale]
    
    # Search for patterns
    move_match = re.search(move_pattern, response, re.IGNORECASE | re.DOTALL)
    rationale_match = re.search(rationale_pattern, response, re.IGNORECASE | re.DOTALL)
    
    # Extract and clean the matches
    best_move = None
    rationale = None
    
    if move_match:
        best_move = move_match.group(1).strip()  # Group 1 contains the move (a number)
        
    if rationale_match:
        rationale = rationale_match.group(2).strip()  # Group 2 contains the rationale content
    
    return best_move, rationale


def parse_blackjack_state(input_text):
    """
    Parse a multi-line blackjack game state text and extract the final state.
    
    Args:
        input_text (str): Multi-line string containing game state progression
    
    Returns:
        str: Formatted string describing the final game state
    """
    # Split the input into lines
    lines = input_text.strip().split('\n')
    
    # Find the last line containing the game state
    game_state_line = None
    for line in lines:
        if line.startswith('Non-Ace Total:'):
            game_state_line = line
    
    if not game_state_line:
        return "No valid game state found"
    
    # Extract the numbers using string split
    parts = game_state_line.split()
    
    # Get the non-ace totals (index 2 and 3 after split)
    player_points = int(parts[2])
    dealer_points = int(parts[3])
    
    # Get the number of aces (index 8 and 9 after split)
    player_aces = int(parts[6])
    dealer_aces = int(parts[7].rstrip(','))
    
    return (f"Player has {player_points} points and {player_aces} aces.\n"
            f"Dealer has {dealer_points} points and {dealer_aces} aces.")


def read_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    return ''.join(lines)

def gemini_generate(prompt, instruction, model_name):
    # the model name can be gemini-2.5-pro-preview-03-25
    client = genai.Client(
        vertexai=True,
        project="zifengw-research",
        location="us-central1",
    )


    model = model_name # "gemini-2.0-flash-001"
    contents = [
        types.Content(role="system", parts=[types.Part.from_text(text=instruction)]),
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)]
    )
    ]
    generate_content_config = types.GenerateContentConfig(
      temperature = 0.7,
      top_p = 0.9,
      max_output_tokens = 100000,
      response_modalities = ["TEXT"],
      safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
      ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
      ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
      ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
      )],
    )

    response = client.models.generate_content(
          model = model,
          contents = contents,
          config = generate_content_config,
      ).text

    return response


def claude_generate(prompt, model_name):
    """
    Generate content using the Claude model.
    """
    client = AnthropicVertex(region='us-east5',
                             project_id="zifengw-research") # Use the appropriate model name here, e.g., 'claude-3-opus-20240229-v1'

    message = client.messages.create(
        model=model_name,
        messages=[{"role": "user","content": prompt}],
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9
    )

    return message.content[0].text  # Return the generated content from Claude

if __name__ == "__main__":
  # Example prompt to test Claude
  prompt = "What is the capital of France?"
  model_name = "claude-3-haiku@20240307"  # Replace with the desired Claude model name

  try:
    # Generate response using Claude
    response = claude_generate(prompt, model_name)
    print("Claude Response:")
    print(response)
  except Exception as e:
    print(f"An error occurred: {e}")