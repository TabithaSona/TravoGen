import os
import gradio as gr
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from huggingface_hub import HfApi


class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str

# Define the LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
    ("human", "Create an itinerary for my day trip."),
])

def create_itinerary(city: str, interests: str) -> str:
    # Process interests
    interests_list = [interest.strip() for interest in interests.split(",")]
    
    # Generate itinerary
    response = llm.invoke(
        itinerary_prompt.format_messages(
            city=city, 
            interests=', '.join(interests_list)
        )
    )
    
    return response.content

# Define the Gradio application
def travel_planner(city: str, interests: str):
    # Generate the itinerary
    itinerary = create_itinerary(city, interests)
    return itinerary

# Build the gradio interface
interface = gr.Interface(
    fn=travel_planner,
    theme='Yntec/HaleyCH_Theme_Orange_Green',
    inputs=[
        gr.Textbox(label="Enter the city for your day trip"),
        gr.Textbox(label="Enter your interests (comma-separated)"),
    ],
    outputs=gr.Textbox(label="Generated Itinerary"),
    title="Travel Itinerary Planner",
    description="Enter a city and your interests to generate a personalized day trip itinerary"
)

# Launch the gradio application
interface.launch(share=False)
