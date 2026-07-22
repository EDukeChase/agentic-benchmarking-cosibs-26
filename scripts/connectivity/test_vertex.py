import os
from google import genai
from google.genai import types

project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

client = genai.Client(
    vertexai=True,
    project=project_id,
    location=location,
)

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=(
        "Use Google Search to find one recently published machine-learning "
        "model for longitudinal electronic health record prediction. "
        "Return its name and source-code repository URL."
    ),
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                google_search=types.GoogleSearch()
            )
        ],
        temperature=0,
    ),
)

print("Response:")
print(response.text)

candidate = response.candidates[0]
grounding = candidate.grounding_metadata

print("\nGrounding metadata:")
print(grounding)

if grounding is None:
    print("\nFAILED: Gemini responded, but no Google Search grounding was recorded.")
else:
    queries = getattr(grounding, "web_search_queries", None)
    chunks = getattr(grounding, "grounding_chunks", None)

    print("\nSearch queries:")
    print(queries)

    print("\nGrounding sources:")
    if chunks:
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web:
                print(f"- {getattr(web, 'title', '')}")
                print(f"  {getattr(web, 'uri', '')}")

    if queries or chunks:
        print("\nSUCCESS: Gemini used Google Search grounding.")
    else:
        print("\nWARNING: Grounding metadata exists but contains no search evidence.")