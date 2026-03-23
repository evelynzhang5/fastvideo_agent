from core.agent_loop import run

def run_agent(question: str) -> str:
    # call your agent loop
    result = run(question)
    
    # IMPORTANT: return a string (for evaluation)
    return str(result)