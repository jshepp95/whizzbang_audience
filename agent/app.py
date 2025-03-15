from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dialogue_manager import get_initial_state, create_workflow
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

# Create workflow once at startup
workflow = create_workflow()
workflow.get_graph().draw_mermaid_png(output_file_path="audience_builder.png")
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str
    conversation_id: str | None = None

# In-memory storage for conversations
conversations = {}

@app.get("/chat/start")
async def start_chat():
    """Start a new chat with a greeting"""
    conversation_id = str(len(conversations) + 1)
    
    # Initialize state
    state = get_initial_state()
    
    # Run the workflow to get greeting
    result = workflow.invoke(state)
    conversations[conversation_id] = result
    
    return {
        "response": result["conversation_history"][-1].content,
        "conversation_id": conversation_id
    }

@app.post("/chat")
async def chat_endpoint(message: Message):
    if not message.conversation_id or message.conversation_id not in conversations:
        return {"error": "Invalid conversation ID. Please start a new chat."}
    
    # Get current state
    current_state = conversations[message.conversation_id]
    
    # Add the user message
    current_state["conversation_history"].append(
        HumanMessage(content=message.message)
    )
    
    # Process the message through workflow
    result = workflow.invoke(current_state)
    
    # Store the updated state
    conversations[message.conversation_id] = result
    
    # Return only the last AI message
    last_message = result["conversation_history"][-1].content
    
    return {
        "response": last_message,
        "conversation_id": message.conversation_id
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)