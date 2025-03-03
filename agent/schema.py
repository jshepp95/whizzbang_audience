from typing import Annotated, List, Optional, Union, TypedDict, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage


class AudienceBuilderState(TypedDict):
    conversation_history: Annotated[List[Union[HumanMessage, AIMessage, Dict]], "conversation history", "append"]
    current_product: Annotated[Optional[str], "current product"]
    product_sku: Annotated[Optional[str], "product sku"]
    product_category: Annotated[Optional[str], "product category"]
    buyer_category: Optional[str]
    current_node: str

class ProductIdentification(BaseModel):
    """Schema for initial product identification from user message"""
    sku: Optional[str] = Field(None, description="The product SKU")
    mentioned: bool = Field(..., description="Whether a product was mentioned in the message")

class ProductDetails(BaseModel):
    """Schema for product details after database lookup"""
    sku: str = Field(..., description="The product SKU")
    product_name: str = Field(None, description="The identified product name")
    buyer_category: Optional[str] = Field(None, description="The buyer category (L4)")
    product_category: Optional[str] = Field(None, description="The product category (L5)")