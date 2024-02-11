from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

from langchain.agents import AgentExecutor, LLMSingleActionAgent, create_openai_tools_agent
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models.llms import create_base_retry_decorator
from litellm import acompletion
from pydantic import Field

from salesgpt.chains import SalesConversationChain, StageAnalyzerChain
from salesgpt.logger import time_logger
from salesgpt.parsers import SalesConvoOutputParser
from salesgpt.prompts import SALES_AGENT_TOOLS_PROMPT
from salesgpt.stages import CONVERSATION_STAGES
from salesgpt.templates import CustomPromptTemplateForTools
from salesgpt.tools import get_tools, setup_knowledge_base

def _create_retry_decorator(llm: Any) -> Callable[[Any], Any]:
    import openai
    errors = [openai.Timeout, openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError]
    return create_base_retry_decorator(error_types=errors, max_retries=llm.max_retries)

class SalesGPT(Chain):
    conversation_history: List[str] = []
    conversation_stage_id: str = "1"
    current_conversation_stage: str = CONVERSATION_STAGES.get("1")
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_agent_executor: Union[AgentExecutor, None] = Field(default=None)
    knowledge_base: Union[RetrievalQA, None] = Field(default=None)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)
    conversation_stage_dict: Dict = CONVERSATION_STAGES

    model_name: str = "gpt-3.5-turbo-0613"
    use_tools: bool = True
    salesperson_name: str = "Ted Lasso"
    salesperson_role: str = "Business Development Representative"
    company_name: str = "Sleep Haven"
    company_business: str = "Sleep Haven is a premium mattress company..."
    company_values: str = "Our mission at Sleep Haven is to help people achieve a better night's sleep..."
    conversation_purpose: str = "find out whether they are looking to achieve better sleep via buying a premier mattress."
    conversation_type: str = "call"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.use_tools:
            self.product_catalog = kwargs.get('product_catalog', 'default_catalog_path.txt')
            self.knowledge_base = setup_knowledge_base(self.product_catalog)
            self.tools = get_tools(self.knowledge_base)
            self.initialize_tool_executor()

    def initialize_tool_executor(self):
        if self.use_tools:
            self.sales_agent_executor = self.setup_with_tools()
        else:
            self.sales_agent_executor = None

    def setup_with_tools(self) -> AgentExecutor:
        tools = get_tools(self.knowledge_base)
        llm_chain = LLMChain(llm=self.llm, prompt=CustomPromptTemplateForTools(...), verbose=self.verbose)
        tool_names = [tool.name for tool in tools]
        sales_agent_with_tools = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=SalesConvoOutputParser(ai_prefix=self.salesperson_name),
            allowed_tools=tool_names,
        )
        return AgentExecutor.from_agent_and_tools(agent=sales_agent_with_tools, tools=tools, verbose=self.verbose)

    @time_logger
    def seed_agent(self):
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    @time_logger
    def determine_conversation_stage(self):
        self.conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history="\n".join(self.conversation_history).rstrip("\n"),
            conversation_stage_id=self.conversation_stage_id,
            conversation_stages="\n".join(
                [str(key) + ": " + str(value) for key, value in CONVERSATION_STAGES.items()]
            ),
        )
        self.current_conversation_stage = self.retrieve_conversation_stage(self.conversation_stage_id)

    def human_step(self, human_input):
        human_input = "User: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)

    @time_logger
    def step(self, stream: bool = False):
        if not stream:
            self._call(inputs={})
        else:
            return self._streaming_generator()

    @time_logger
    def astep(self, stream: bool = False):
        if not stream:
            self._acall(inputs={})
        else:
            return self._astreaming_generator()

    @time_logger
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.sales_agent_executor is None:
            print("sales_agent_executor is not initialized.")
            return {}
        inputs = {
            "input": "",
            "conversation_stage": self.current_conversation_stage,
            "conversation_history": "\n".join(self.conversation_history),
            "salesperson_name": self.salesperson_name,
            "salesperson_role": self.salesperson_role,
            "company_name": self.company_name,
            "company_business": self.company_business,
            "company_values": self.company_values,
            "conversation_purpose": self.conversation_purpose,
            "conversation_type": self.conversation_type,
        }
        ai_message = self.sales_agent_executor.invoke(inputs)
        output = ai_message["output"]
        agent_response = f"{self.salesperson_name}: {output} <END_OF_TURN>"
        self.conversation_history.append(agent_response)
        print(agent_response.replace("<END_OF_TURN>", ""))
        return ai_message

    @classmethod
    @time_logger
    def from_llm(cls, llm: ChatLiteLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
        # This method remains largely as provided, ensuring tool setup is included if use_tools is True.

# Ensure all placeholders like "..." are replaced with actual implementation details specific to your application.
    @classmethod
    @time_logger
    def from_llm(cls, llm: ChatLiteLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
        # Initialize stage analyzer and sales conversation chains based on the provided LLM model
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)
        sales_conversation_utterance_chain = SalesConversationChain.from_llm(llm, verbose=verbose)

        # Check if tools should be used and set up accordingly
        if kwargs.get("use_tools", False):
            # If specific tool-related configurations are provided, use them to set up tools and knowledge base
            product_catalog = kwargs.get("product_catalog", "default_catalog_path.txt")
            knowledge_base = setup_knowledge_base(product_catalog)
            tools = get_tools(knowledge_base)

            # Set up the executor with tools, providing necessary configurations
            sales_agent_executor = cls.setup_executor_with_tools(llm=llm, tools=tools, verbose=verbose, **kwargs)
        else:
            # If tools are not used, the executor can be None or set up differently as per application requirements
            sales_agent_executor = None

        # Return a new instance of SalesGPT with all components initialized
        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            model_name=llm.model_name,
            verbose=verbose,
            **kwargs
        )

    @staticmethod
    def setup_executor_with_tools(llm: ChatLiteLLM, tools: List[Any], verbose: bool = False, **kwargs) -> AgentExecutor:
        # Customize this method to set up the AgentExecutor with the tools.
        # This is a placeholder. The actual implementation depends on how tools are used within your application.
        tool_names = [tool.name for tool in tools]
        custom_prompt_template = CustomPromptTemplateForTools(
            template=SALES_AGENT_TOOLS_PROMPT,
            tools_getter=lambda: tools,
            # Add other necessary parameters for the prompt template
        )
        llm_chain = LLMChain(llm=llm, prompt=custom_prompt_template, verbose=verbose)
        output_parser = SalesConvoOutputParser(ai_prefix="SalesGPT")  # Customize as needed

        sales_agent_with_tools = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            allowed_tools=tool_names,
            # Additional configurations as required
        )

        return AgentExecutor.from_agent_and_tools(
            agent=sales_agent_with_tools,
            tools=tools,
            verbose=verbose
        )

