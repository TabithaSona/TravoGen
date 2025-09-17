import os
import gradio as gr
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from huggingface_hub import HfApi

# Define the LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Updated prompt to include exclusions
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}.
     
EXCLUSION REQUIREMENTS:
- Do NOT include any of these places: {exclusions}
- If the user mentions general exclusions like "crowded places" or "expensive restaurants", interpret them appropriately
- Ensure the itinerary completely avoids the excluded places

Provide a brief, bulleted itinerary with time slots for each activity."""),
    ("human", "Create an itinerary for my day trip excluding the specified places."),
])

def create_itinerary(city: str, interests: str, exclusions: str = "") -> str:
    # Process interests
    interests_list = [interest.strip() for interest in interests.split(",") if interest.strip()]
    
    # Process exclusions
    exclusions_list = [exclusion.strip() for exclusion in exclusions.split(",") if exclusion.strip()]
    
    # Format exclusions for the prompt
    if exclusions_list:
        formatted_exclusions = ", ".join(exclusions_list)
    else:
        formatted_exclusions = "none"
    
    # Generate itinerary
    response = llm.invoke(
        itinerary_prompt.format_messages(
            city=city, 
            interests=', '.join(interests_list),
            exclusions=formatted_exclusions
        )
    )
    
    return response.content

def travel_planner(city: str, interests: str, exclusions: str):
    # Generate the itinerary with exclusions
    itinerary = create_itinerary(city, interests, exclusions)
    return itinerary

# Build the gradio interface with enhanced inputs
with gr.Blocks(theme="gstaff/sketch", title="Travel Itinerary Planner") as interface:
    gr.Markdown("# 🗺️ Travel Itinerary Planner")
    gr.Markdown("Enter a city and your interests to generate a personalized day trip itinerary. You can exclude specific places you don't want to visit.")
    
    with gr.Row():
        with gr.Column():
            city_input = gr.Textbox(
                label="🏙️ Enter the city for your day trip",
                placeholder="e.g., Paris, Tokyo, New York..."
            )
            
            interests_input = gr.Textbox(
                label="🎯 Enter your interests (comma-separated)",
                placeholder="e.g., museums, parks, local cuisine, shopping..."
            )
            
            exclusions_input = gr.Textbox(
                label="🚫 Places to exclude (comma-separated)",
                placeholder="e.g., crowded markets, expensive restaurants, specific museum names...",
                value=""
            )
            
            submit_btn = gr.Button("Generate Itinerary", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="📋 Generated Itinerary",
                lines=15,
                interactive=False
            )
    
    # Examples for users
    gr.Examples(
        examples=[
            ["Paris", "art, architecture, cafes", "Louvre Museum, crowded places"],
            ["Tokyo", "technology, temples, food", "expensive restaurants, amusement parks"],
            ["New York", "museums, parks, broadway", "Times Square, crowded areas"]
        ],
        inputs=[city_input, interests_input, exclusions_input],
        label="💡 Try these examples:"
    )
    
    submit_btn.click(
        fn=travel_planner,
        inputs=[city_input, interests_input, exclusions_input],
        outputs=output
    )

# Launch the gradio application
if __name__ == "__main__":
    interface.launch()
