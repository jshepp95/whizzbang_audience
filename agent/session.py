import os
import json
from dotenv import load_dotenv
from typing import Dict, Optional
from uuid import uuid4
import logging
from redis import Redis

from dialogue_manager import get_initial_state, create_workflow

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")

os.makedirs("agent/logs", exist_ok=True)

# logging.basicConfig(level=logging.INFO, filename="agent/logs/session.log")
# logger = logging.getLogger(__name__)

# TODO: Use Langchain Output Parsers (JSON)


class SessionManager:
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.session_ttl = 3600
        self.workflow = create_workflow()
        
    def create_session(self) -> str:
        session_id = str(uuid4())
        initial_state = get_initial_state()
        self.save_state(session_id, initial_state)
        # logger.info(f"Created new session: {session_id}")
        # logger.info(f"Initial state: {initial_state}")

        return session_id
    
    def get_state(self, session_id: str) -> Optional[Dict]:
        state_json = self.redis.get(f"session:{session_id}")
        state = json.loads(state_json) if state_json else None
        # logger.info(f"Retrieved state for session {session_id}: {state}")

        return json.loads(state_json) if state_json else None
    
    def save_state(self, session_id: str, state: Dict) -> None:
        # Convert state to a JSON-serializable format
        state_copy = state.copy()
        
        # Extract just the conversation data we need
        conversation_history = []
        for msg in state_copy["conversation_history"]:
            try:
                # Try to access as an object
                role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                content = msg.content
            except AttributeError:
                # Handle as a dictionary
                role = msg.get("role")
                content = msg.get("content")
                
            conversation_history.append({
                "role": role,
                "content": content
            })
        
        state_copy["conversation_history"] = conversation_history
        
        self.redis.setex(
            f"session:{session_id}",
            self.session_ttl,
            json.dumps(state_copy)
        )