import base64
import pathlib
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
from pathlib import Path

load_dotenv("API.env")

# Load existing results
output_file = "emotion_results.json"
results = {}
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} previously processed files\n")

# Load all available API keys
api_keys = []
i = 1
while True:
    key_name = f"MERALION_API_KEY_{i}" if i > 1 else "MERALION_API_KEY"
    key = os.getenv(key_name)
    if not key:
        break
    api_keys.append(key)
    i += 1

if not api_keys:
    print("Error: No API keys found in API.env")
    exit(1)

print(f"Found {len(api_keys)} API key(s)\n")

# Directory containing WAV files
wav_dir = Path("wav")
all_files = sorted(wav_dir.glob("*.wav"))
total_files = len(all_files)

current_key_index = 0

# Loop through all WAV files in the directory
for i, audio_file in enumerate(all_files, 1):
    # Skip already processed files
    if audio_file.name in results:
        print(f"({i}/{total_files}) Skipping already processed: {audio_file.name}")
        continue
    
    audio_path = str(audio_file)
    print(f"Processing ({i}/{total_files}): {audio_path}")
    
    audio_b64 = base64.b64encode(pathlib.Path(audio_path).read_bytes()).decode()
    
    # Try with current API key, switch if needed
    max_retries = len(api_keys)
    retry_count = 0
    success = False
    
    while retry_count < max_retries and not success:
        try:
            current_key = api_keys[current_key_index]
            print(f"Using API key {current_key_index + 1}/{len(api_keys)}")
            
            client = OpenAI(
                api_key=current_key,
                base_url="http://meralion.org:8010/v1",
            )
            
            response = client.chat.completions.create(
                model="MERaLiON/MERaLiON-3-10B",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Listen to this audio clip and classify the emotion of the speaker. "
                                "Choose from: anger, anxiety, boredom, disgust, fear, happiness, neutral, sadness. "
                                "Reply with just the emotion label."
                            )
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}
                        },
                    ],
                }],
                max_tokens=10,
            )
            
            emotion = response.choices[0].message.content.strip()
            results[audio_file.name] = emotion
            print(f"File : {audio_path}")
            print(f"Emotion: {emotion}\n")
            
            # Save results after each file (in case of error)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            success = True
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error with API key {current_key_index + 1}: {error_msg}")
            
            # Switch to next API key (without wrapping)
            if current_key_index < len(api_keys) - 1:
                current_key_index += 1
            retry_count += 1
            
            if retry_count < max_retries:
                print(f"Switching to API key {current_key_index + 1}/{len(api_keys)}\n")
                time.sleep(2)  # Brief delay before retry
    
    if not success:
        print(f"Failed to process {audio_file.name} with all API keys. Skipping.\n")
    else:
        # Rate limit: 5 executions per minute = 12 seconds per request
        remaining_files = total_files - i
        if remaining_files > 0:
            print("Waiting 12 seconds to respect rate limit (5 requests/minute)...")
            time.sleep(12)

print(f"\nResults saved to {output_file}")
print(f"Total files processed: {len(results)}/{total_files}")