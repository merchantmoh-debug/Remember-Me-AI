from typing import Any, Dict, List, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
try:
    from langchain.memory import BaseMemory
except ImportError:
    try:
        from langchain.schema import BaseMemory
    except ImportError:
        # Fallback for newer modular versions or if not found
        from pydantic import BaseModel
        class BaseMemory(BaseModel):
            pass

from pydantic import Field
from ..core.csnp import CSNPManager

class CSNPLangChainMemory(BaseMemory):
    """
    LangChain Adapter for Remember Me AI's CSNP Kernel.

    Replaces standard ConversationBufferMemory with a Wasserstein-Optimal,
    Merkle-Verified memory stream.

    Usage:
        memory = CSNPLangChainMemory(context_limit=10)
        chain = ConversationChain(llm=llm, memory=memory)
    """
    class Config:
        arbitrary_types_allowed = True

    csnp: CSNPManager = Field(default_factory=lambda: CSNPManager(context_limit=10))
    memory_key: str = "history"
    input_key: Optional[str] = None
    output_key: Optional[str] = None

    def __init__(self, context_limit: int = 10, **kwargs):
        super().__init__(**kwargs)
        # Initialize CSNP if not passed in explicitly
        if "csnp" not in kwargs:
            self.csnp = CSNPManager(context_limit=context_limit)

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves the Coherent State from the CSNP Kernel.
        """
        # Get the compressed, verified context
        context_str = self.csnp.retrieve_context()

        # Format as string (standard for basic Prompts)
        return {self.memory_key: context_str}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Injects the interaction into the Living State Vector.
        """
        input_str, output_str = self._get_input_output(inputs, outputs)

        # Update the Coherent State (Embed -> Evolve -> Compress)
        self.csnp.update_state(input_str, output_str)

    def clear(self) -> None:
        """
        Resets the kernel.
        """
        # Re-initialize the manager
        self.csnp = CSNPManager(context_limit=self.csnp.context_limit)

    def _get_input_output(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> tuple[str, str]:
        if self.input_key:
            input_str = inputs[self.input_key]
        elif len(inputs) == 1:
            input_str = list(inputs.values())[0]
        else:
            raise ValueError("Input key not set and multiple inputs found.")

        if self.output_key:
            output_str = outputs[self.output_key]
        elif len(outputs) == 1:
            output_str = list(outputs.values())[0]
        else:
            # Try to find a likely output key
            if "response" in outputs:
                output_str = outputs["response"]
            elif "text" in outputs:
                output_str = outputs["text"]
            else:
                 raise ValueError("Output key not set and multiple outputs found.")

        return str(input_str), str(output_str)
