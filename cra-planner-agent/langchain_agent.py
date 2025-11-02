import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub
from langchain_core.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Any, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

# Load environment variables
load_dotenv()

class GPT5NanoWrapper(BaseChatModel):
    """Wrapper for gpt-5-nano that strips unsupported parameters."""

    llm: AzureChatOpenAI

    class Config:
        arbitrary_types_allowed = True

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        # Remove unsupported parameters
        kwargs.pop('stop', None)
        # Call the underlying LLM without stop parameter
        return self.llm._generate(messages, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "gpt-5-nano-wrapper"

    def bind_tools(self, tools, **kwargs):
        return self.llm.bind_tools(tools, **kwargs)

# Sample tools for the agent
def get_current_time(query: str) -> str:
    """Returns the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculator(query: str) -> str:
    """Performs basic arithmetic calculations. Input should be a math expression."""
    try:
        result = eval(query)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def search_info(query: str) -> str:
    """Simulates searching for information. Replace with actual search API if needed."""
    return f"Here's some information about: {query}"

# Define tools
tools = [
    Tool(
        name="GetCurrentTime",
        func=get_current_time,
        description="Useful for getting the current date and time"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for performing mathematical calculations. Input should be a valid mathematical expression."
    ),
    Tool(
        name="SearchInfo",
        func=search_info,
        description="Useful for searching information about a topic"
    )
]

def create_langchain_agent():
    """Create and return a LangChain agent configured with Azure OpenAI."""

    # Get configuration from environment
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not all([api_key, endpoint, deployment]):
        raise ValueError("Missing required environment variables. Please check your .env file.")

    # Initialize the AzureChatOpenAI model with proper Azure configuration
    # Note: gpt-5-nano model has limited parameter support
    # - Only supports temperature=1 (default)
    # - Does not support 'stop' parameter
    # - Uses 'max_completion_tokens' instead of 'max_tokens'
    base_llm = AzureChatOpenAI(
        azure_deployment=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version
    )

    # Wrap the model to strip unsupported parameters
    llm = GPT5NanoWrapper(llm=base_llm)

    # Pull the ReAct prompt from LangChain hub and customize it
    # The default prompt doesn't work well with reasoning models that tend to
    # generate the entire expected conversation including observations
    from langchain_core.prompts import PromptTemplate

    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action

IMPORTANT: After writing "Action Input:", STOP immediately. Do NOT write "Observation:" - that will be provided to you after the tool executes.

Observation: the result of the action (this will be provided by the system)
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

    return agent_executor

def main():
    """Main function to run the agent."""
    print("Initializing LangChain Agent with Azure OpenAI...")

    try:
        agent_executor = create_langchain_agent()
        print("Agent initialized successfully!\n")

        # Example queries
        queries = [
            "What is the current time?",
            "Calculate 25 * 4 + 10",
            "Tell me about artificial intelligence"
        ]

        print("Running example queries...\n")
        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            result = agent_executor.invoke({"input": query})
            print(f"\nFinal Answer: {result['output']}\n")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease ensure:")
        print("1. You have created a .env file with the required variables")
        print("2. Your Azure OpenAI API key is valid")
        print("3. The deployment name and endpoint are correct")

if __name__ == "__main__":
    main()
