import os
import logging
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import AzureChatOpenAI

from schema import AudienceBuilderState, ProductIdentification
from tools import SKULookupTool

from pprint import pprint

load_dotenv()

# os.makedirs("logs/", exist_ok=True)

# logging.basicConfig(level=logging.INFO, filename=f"agent/logs/{__name__}.log")
# logger = logging.getLogger(__name__)

# TODO: Input-Output Schema Nodes
# TODO: Search SKU Tool

AZURE_OAI_KEY = os.getenv("AZURE_OAI_KEY")
END_POINT = os.getenv("END_POINT")
DEPLOYMENT_NAME = "gpt-4o"
API_VERSION_GPT = os.getenv("API_VERSION_GPT")

llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    openai_api_version=API_VERSION_GPT,
    azure_endpoint=END_POINT,
    api_key=AZURE_OAI_KEY,
    temperature=0
)

def greet(state: AudienceBuilderState) -> AudienceBuilderState:
    pprint(f"Greeting user from state: {state}")
    
    prompt = ChatPromptTemplate.from_messages([
        """You are an audience building assistant for Retail Media.
        Greet the user warmly and ask which product and correspondingSKU they'd like to build audiences for
        """]
    )

    chain = prompt | llm
    response = chain.invoke({})

    # print(response.content)

    return {
        **state,
        "conversation_history": state["conversation_history"] + [
            AIMessage(content=response.content)
        ],
        "current_node": "identify_product"
    }


def identify_product(state: AudienceBuilderState) -> AudienceBuilderState:

    pprint(f"Identifying product from state: {state}")
    
    messages = state["conversation_history"]
    
    last_user_message = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            last_user_message = message.content
            break

    if not last_user_message:
        return {**state, "current_node": END}
    
    parser = PydanticOutputParser(pydantic_object=ProductIdentification)
    
    prompt = ChatPromptTemplate.from_messages([
        """Extract the product SKU that the user wants to build audiences for.
        
        User Message: {user_message}

        {format_instructions}
        """
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "user_message": last_user_message,
        "format_instructions": parser.get_format_instructions()
        })

    # TODO: Should we overaload the node with multiple prompts, or should we have a separate node for each prompt?
    if not result.sku:
        clarification_prompt = ChatPromptTemplate.from_template(
            """I'm not sure which product you'd like to build audiences for. 
            Could you please specify the product SKU?"""
        )
        
        clarification_chain = clarification_prompt | llm
        clarification_response = clarification_chain.invoke({})

        return {
            **state,
            "conversation_history": state["conversation_history"] + [
                AIMessage(content=clarification_response.content)
            ],
            "current_node": "identify_product"
        }

    else:
        confirmation_prompt = ChatPromptTemplate.from_template(
            """
            You are an audience building assistant for retail media.

            The user wants to build audiences for the product: {sku}

            Respond with a brief, friendly confirmation that you'll help them build audiences for this product.

            """
        ) 

        confirmation_chain = confirmation_prompt | llm
        confirmation_response = confirmation_chain.invoke({
            "sku": result.sku
        })
        
        return {
            **state,
            "sku": result.sku,
            # "current_node": "lookup_product_details",
            "current_node": END,
            "conversation_history": state["conversation_history"] + [
                AIMessage(content=confirmation_response.content)
            ]
        }
    
def lookup_product_details(state: AudienceBuilderState) -> AudienceBuilderState:

    print(f"Looking up product details for SKU: {state.get('sku')}")

    product_sku = state.get("sku")

    sku_lookup_tool = SKULookupTool()

    try:
        product_details = sku_lookup_tool.invoke(product_sku)

        response_prompt = ChatPromptTemplate.from_template(
            """ You are an audience building assistant for retail media 
            
            You just recieved the details for the Product SKU {sku}:
            - Product Name: {product_name}
            - Product Category (L5): {product_category}
            - Buyer Category: {buyer_category}

            Respond to the user with a friendly message confirming these details.
            and letting them know you'll search the data and now suggest audience segments based on these details.
            
            """
        )

        response_chain = response_prompt | llm
        response = response_chain.invoke({
            "sku": product_details.sku,
            "product_name": product_details.product_name,
            "product_category": product_details.product_category,
            "buyer_category": product_details.buyer_category
        })

        return {
            **state,
            "product_name": product_details.product_name,
            "product_category": product_details.product_category,
            "buyer_category": product_details.buyer_category,
            "current_node": END,
            "conversation_history": state["conversation_history"] + [
                AIMessage(content=response.content)
            ]
        }
    
    except Exception as e:
        not_found_prompt = ChatPromptTemplate.from_template(
            """You are an audience building assistant for retail media.
            
            The user asked about SKU {sku}, but it could not be found in our database.
            
            Politely inform them that you couldn't find this SKU and ask if they'd like to try a different product SKU.
            """
        )

        not_found_chain = not_found_prompt | llm
        not_found_response = not_found_chain.invoke({"sku": product_sku})
        
        return {
            **state,
            "conversation_history": state["conversation_history"] + [
                AIMessage(content=not_found_response.content)
            ],
            "current_node": "identify_product"
        }

def create_workflow():
    workflow = StateGraph(AudienceBuilderState)
    
    workflow.add_node("greet", greet)
    workflow.add_node("identify_product", identify_product)
    # workflow.add_node("lookup_product_details", lookup_product_details)  # Add the new node

    workflow.add_edge("greet", "identify_product")
    
    # Update the conditional edges to include the new path
    workflow.add_conditional_edges(
        "identify_product",
        lambda x: x["current_node"],
        {
            "identify_product": "identify_product",
            # "lookup_product_details": "lookup_product_details",  # Add this line
            END: END
        }
    )
    
    # Add conditional edges for the new node
    # workflow.add_conditional_edges(
    #     "lookup_product_details",
    #     lambda x: x["current_node"],
    #     {
    #         "identify_product": "identify_product",
    #         END: END
    #     }
    # )

    workflow.set_entry_point("greet")
    
    return workflow.compile()

# def create_workflow():
#     workflow = StateGraph(AudienceBuilderState)
    
#     workflow.add_node("greet", greet)
#     workflow.add_node("identify_product", identify_product)

#     workflow.add_edge("greet", "identify_product")
    
#     workflow.add_conditional_edges(
#         "identify_product",
#         lambda x: x["current_node"],
#         {
#             "identify_product": "identify_product",
#             END: END
#         }
#     )

#     workflow.set_entry_point("greet")
    
#     return workflow.compile()

def get_initial_state():
    return {
        "conversation_history": [],
        "sku": None,
        "product_name": None,
        "product_category": None,
        "buyer_category": None,
        "current_node": "greet"
    }




    
# except Exception as e:
#     error_prompt = ChatPromptTemplate.from_template(
#         """You are an audience building assistant for retail media.
        
#         There was a technical issue while looking up information for SKU {sku}.
        
#         Apologize to the user and ask them to try again later or provide a different SKU.
#         """
#     )
    
#     error_chain = error_prompt | llm
#     error_response = error_chain.invoke({"sku": product_sku})
    
#     return {
#         **state,
#         "conversation_history": state["conversation_history"] + [
#             AIMessage(content=error_response.content)
#         ],
#         "current_node": "identify_product"
#     }