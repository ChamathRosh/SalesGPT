import json
from typing import List, Tuple

from langchain_community.chat_models import ChatLiteLLM
from salesgpt.agents import SalesGPT

GPT_MODEL = "gpt-3.5-turbo-0613"

class SalesGPTAPI:
    USE_TOOLS = True  # Ensure tools are enabled

    def __init__(self, config_path: str, verbose: bool = False, max_num_turns: int = 10):
        self.config_path = config_path
        self.verbose = verbose
        self.max_num_turns = max_num_turns
        self.llm = ChatLiteLLM(temperature=0.2, model_name=GPT_MODEL)

    def do(self, conversation_history: List[str], human_input: str = None) -> Tuple[str, str]:
        if self.config_path == "":
            print("No agent config specified, using a standard config")
            sales_agent = SalesGPT.from_llm(
                self.llm,
                use_tools=self.USE_TOOLS,  # Directly pass the USE_TOOLS flag
                product_catalog="examples/sample_product_catalog.txt",  # Specify the product catalog path
                salesperson_name="Ted Lasso",  # Customization for the agent's identity
                verbose=self.verbose
            )
        else:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            if self.verbose:
                print(f"Agent config: {config}")
            sales_agent = SalesGPT.from_llm(self.llm, verbose=self.verbose, **config)

        # Check turns
        current_turns = len(conversation_history) + 1
        if current_turns >= self.max_num_turns:
            print("Maximum number of turns reached - ending the conversation.")
            return "<END_OF_CONVERSATION>", "Maximum number of turns reached."

        # Seed the conversation and manage history
        sales_agent.seed_agent()
        sales_agent.conversation_history = conversation_history

        # Process human input
        if human_input is not None:
            sales_agent.human_step(human_input)

        # Generate agent's reply
        sales_agent.step()

        # Check for conversation end signal
        if "<END_OF_CALL>" in sales_agent.conversation_history[-1]:
            print("Sales Agent determined it is time to end the conversation.")
            return "<END_OF_CALL>", "Sales Agent determined it is time to end the conversation."

        reply = sales_agent.conversation_history[-1]

        if self.verbose:
            print("=" * 10)
            print(f"{sales_agent.salesperson_name}: {reply}")

        # Split the reply to get the agent's name and what they said
        name, response = reply.split(": ", 1)
        return name, response
