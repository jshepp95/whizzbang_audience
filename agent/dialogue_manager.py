import os
import logging
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser
from langchain_openai import AzureChatOpenAI

from schema import AudienceBuilderState, ProductIdentification, ProductSearchResults
from tools import SKULookupTool, ProductLookupTool

from pprint import pprint

load_dotenv()

# os.makedirs("logs/", exist_ok=True)

# logging.basicConfig(level=logging.INFO, filename=f"agent/logs/{__name__}.log")
# logger = logging.getLogger(__name__)

# TODO: Put onto a Production URL 
# TODO: Learn How to Trace Objects and Debug in VSCode
# TODO: Product Name Lookup
# TODO: Table of Results
# TODO: Results Table
# TODO: Checkbox Selection
# TODO: Streaming
# TODO: Handel Edge Cases, non-flow behaviours

AZURE_OAI_KEY = os.getenv("AZURE_OAI_KEY")
END_POINT = os.getenv("END_POINT")
DEPLOYMENT_NAME = "gpt-4o"
API_VERSION_GPT = os.getenv("API_VERSION_GPT")

llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    openai_api_version=API_VERSION_GPT,
    azure_endpoint=END_POINT,
    api_key=AZURE_OAI_KEY,
    temperature=0,
    streaming=True
)

def greet(state: AudienceBuilderState) -> AudienceBuilderState:
    pprint(f"\n\nGreeting user from state: {state}")
    
    prompt = ChatPromptTemplate.from_messages([
        """You are an audience building assistant for Nectar 360.
        Greet the user warmly and ask which product and corresponding SKU they'd like to build audiences for.

        Don't sound cheesy or corporate.
        """]
    )

    chain = prompt | llm
    response = chain.invoke({})

    return {
        **state,
        "conversation_history": state["conversation_history"] + [
            AIMessage(content=response.content)
        ],
        "current_node": "identify_product"
    }

# TODO: Handel Name or SKU
def identify_product(state: AudienceBuilderState) -> AudienceBuilderState:
    pprint(f"\n\nIdentifying product from state: {state}")
    
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
        """Extract the product that the user wants to build audiences for.
        
        User Message: {user_message}

        {format_instructions}
        """
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "user_message": last_user_message,
        "format_instructions": parser.get_format_instructions()
    })

    if not result.product_name:
        # Could not parse a SKU: ask for clarification again
        clarification_prompt = ChatPromptTemplate.from_template(
            """I'm not sure which product you'd like to build audiences for. 
            Could you please specify the product name?"""
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
        # Found a SKU -> confirm and then move to lookup step
        confirmation_prompt = ChatPromptTemplate.from_template(
            """
            You are an audience building assistant for retail media.

            The user wants to build audiences for the product: {product_name}

            Respond with a brief, friendly confirmation that you'll help them build audiences for this product.
            """
        ) 

        confirmation_chain = confirmation_prompt | llm
        confirmation_response = confirmation_chain.invoke({
            "product_name": result.product_name,
        })
        
        return {
            **state,
            "product_name": result.product_name,
            "conversation_history": state["conversation_history"] + [
                AIMessage(content=confirmation_response.content)
            ],
            "current_node": "lookup_product_details"
        }

def lookup_product_details(state: AudienceBuilderState) -> AudienceBuilderState:

    print(f"\n\nLooking up product details for Product Name: {state.get('product_name')}")

    product_name = state.get("product_name")
    product_lookup_tool = ProductLookupTool()

    try:
        product_search_results = product_lookup_tool.invoke(product_name)

        # Summarize the details to the user
        response_prompt = ChatPromptTemplate.from_template(
            """You are an audience building assistant for retail media.
            
            You just received details for the Product Name {product_name}. 
            You have also just run a search to return similar product variants, with results grouped by Buyer Category and Product Categories.
            
            - Product Name: {product_name}
            - Product Details: {product_search_results}

            Respond warmly to the user confirming the Product Name.
            Summarise the product variants that have been found, specifying the unique Buyer Categories and Product Categories.

            Do not say 'Hi' or 'Hello' or anything like that. You have already spoken with the user.

            YOU MUST RESPOND as the assistant.
            """
        )

        response_chain = response_prompt | llm
        response = response_chain.invoke({
            "product_name": product_name,
            "product_search_results": product_search_results
        })

        print(f"\n\nResponse: {response.content}")

        return {
            **state,
            "product_name": product_name,
            "product_search_results": product_search_results,
            "current_node": "format_product_table",
            "conversation_history": state["conversation_history"] + [
                AIMessage(content=response.content)
            ]
        }
    
    except Exception as e:
        print("\nException in lookup_product_details:", repr(e))
        # If the SKU cannot be found or something else goes wrong
        not_found_prompt = ChatPromptTemplate.from_template(
            """You are an audience building assistant for retail media.
            
            The user asked about Product Name {product_name}, but it could not be found in our database.
            
            Politely inform them that you couldn't find this Product and ask if they'd like to try a different product.

            Respond as the assistant.
            """
        )

        not_found_chain = not_found_prompt | llm
        not_found_response = not_found_chain.invoke({"name": product_name})
        
        return {
            **state,
            "conversation_history": state["conversation_history"] + [
                AIMessage(content=not_found_response.content)
            ],
            "current_node": END
        }

# TODO: Mardown Table
# TODO: react-chat-ui-kit, botframework-webchat, stream-chat-react
def format_product_table(state: AudienceBuilderState) -> AudienceBuilderState:
    print("\n\nFormatting Search Results")
    
    # Get the product search results from state
    product_search_results = state.get("product_search_results")
    
    # Create a prompt template that takes the structured data
    response_prompt = ChatPromptTemplate.from_template("""
    You are a data formatter that creates clean, readable summaries from product data.
    
    Here are the search results for products:
    
    Buyer Categories: {buyer_categories}
    Product Categories: {product_categories}
    Total Results: {total_results}
    
    Sample products:
    {sample_products}
    
    1. First, provide a brief summary of the search results.
    2. Then, create a well-formatted markdown table showing the most relevant products.
    3. Include columns for: Buyer Category, Product Category, SKU Number, Product Name.
    4. Limit to showing at most 10 products total.
    """)
    
    # Format the product data for the prompt
    # Convert ProductDetails objects to strings for display in the prompt
    sample_products = []
    for p in product_search_results.all_products[:5]:  # Just show 5 examples
        sample_products.append(
            f"- {p.product_name} (SKU: {p.sku}, Buyer Category: {p.buyer_category}, Product Category: {p.product_category})"
        )
    
    # Create the chain
    response_chain = response_prompt | llm
    
    # Invoke the chain with the formatted product data
    response = response_chain.invoke({
        "buyer_categories": ", ".join(product_search_results.unique_buyer_categories),
        "product_categories": ", ".join(product_search_results.unique_product_categories),
        "total_results": product_search_results.total_results,
        "sample_products": "\n".join(sample_products)
    })
    
    # Extract the content from the response
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = response
    
    # Return updated state with the formatted response
    return {
        **state,
        "conversation_history": state["conversation_history"] + [
            AIMessage(content=content)
        ],
        "current_node": END
    }

def create_workflow():
    workflow = StateGraph(AudienceBuilderState)
    
    workflow.add_node("greet", greet)
    workflow.add_node("identify_product", identify_product)
    workflow.add_node("lookup_product_details", lookup_product_details)
    workflow.add_node("format_product_table", format_product_table)
    
    # Edge: greet -> identify_product
    workflow.add_edge("greet", "identify_product")
    
    # Edge from identify_product depends on its "current_node" output
    workflow.add_conditional_edges(
        "identify_product",
        lambda x: x["current_node"],
        {
            "identify_product": "identify_product",
            "lookup_product_details": "lookup_product_details",
            END: END
        }
    )

    # Now set up the edges for lookup_product_details
    workflow.add_conditional_edges(
        "lookup_product_details",
        lambda x: x["current_node"],
        {
            "identify_product": "identify_product",
            "format_product_table": "format_product_table",
            END: END
        }
    )

    workflow.set_entry_point("greet")
    
    return workflow.compile()

def get_initial_state():
    return {
        "conversation_history": [],
        # "sku": None,
        "product_name": None,
        "product_category": None,
        "buyer_category": None,
        "product_search_results": None,
        "current_node": "greet",
        
        
    }

# def format_product_table(state: AudienceBuilderState):

#     print(f"\n\nFormatting Search Results")

#     search_results = state.get("product_search_results")
#     print(search_results)

#     prompt = f"""
#     Based on the search for "{search_results}", format the following products into a markdown table:
    
#     {[p.model_dump() for p in search_results.all_products[:10]]}
    
#     Include columns for SKU, Product Name, Buyer Category, and Product Category.
#     Format the data as a clean markdown table with aligned columns.
#     """
    
#     formatted_table = llm.invoke(prompt)
    
#     return {
#         **state,
#         "conversation_history": state["conversation_history"] + [
#             AIMessage(content=formatted_table.content)
#         ],
#         "current_node": END
#     }


# def identify_product(state: AudienceBuilderState) -> AudienceBuilderState:
#     pprint(f"\n\nIdentifying product from state: {state}")
    
#     messages = state["conversation_history"]
    
#     last_user_message = None
#     for message in reversed(messages):
#         if isinstance(message, HumanMessage):
#             last_user_message = message.content
#             break

#     if not last_user_message:
#         return {**state, "current_node": END}
    
#     parser = PydanticOutputParser(pydantic_object=ProductIdentification)
    
#     prompt = ChatPromptTemplate.from_messages([
#         """Extract the product SKU that the user wants to build audiences for.
        
#         User Message: {user_message}

#         {format_instructions}
#         """
#     ])

#     chain = prompt | llm | parser
#     result = chain.invoke({
#         "user_message": last_user_message,
#         "format_instructions": parser.get_format_instructions()
#     })

#     if not result.sku:
#         # Could not parse a SKU: ask for clarification again
#         clarification_prompt = ChatPromptTemplate.from_template(
#             """I'm not sure which product you'd like to build audiences for. 
#             Could you please specify the product SKU?"""
#         )
        
#         clarification_chain = clarification_prompt | llm
#         clarification_response = clarification_chain.invoke({})

#         return {
#             **state,
#             "conversation_history": state["conversation_history"] + [
#                 AIMessage(content=clarification_response.content)
#             ],
#             "current_node": "identify_product"
#         }

#     else:
#         # Found a SKU -> confirm and then move to lookup step
#         confirmation_prompt = ChatPromptTemplate.from_template(
#             """
#             You are an audience building assistant for retail media.

#             The user wants to build audiences for the product: {sku}

#             Respond with a brief, friendly confirmation that you'll help them build audiences for this product.
#             """
#         ) 

#         confirmation_chain = confirmation_prompt | llm
#         confirmation_response = confirmation_chain.invoke({
#             "sku": result.sku
#         })
        
#         return {
#             **state,
#             "sku": result.sku,
#             "conversation_history": state["conversation_history"] + [
#                 AIMessage(content=confirmation_response.content)
#             ],
#             "current_node": "lookup_product_details"
#         }

# def lookup_product_details(state: AudienceBuilderState) -> AudienceBuilderState:

#     print(f"\n\nLooking up product details for SKU: {state.get('sku')}")

#     product_sku = state.get("sku")
#     sku_lookup_tool = SKULookupTool()

#     try:
#         product_details = sku_lookup_tool.invoke(product_sku)

#         # Summarize the details to the user
#         response_prompt = ChatPromptTemplate.from_template(
#             """You are an audience building assistant for retail media.
            
#             You just received details for the Product SKU {sku}:
#             - Product Name: {product_name}
#             - Product Category: {product_category}
#             - Buyer Category: {buyer_category}

#             Respond warmly to the user confirming the Product Name and the SKU.

#             Do not say 'Hi' or 'Hello' or anything like that. You have already spoken with the user.

#             Recommend to build audiences for the given Product Category and the Buyer Category.

#             Then close by asking if they'd like to proceed with building audiences for this product.

#             YOU MUST RESPOND as the assistant.
#             """
#         )

#         response_chain = response_prompt | llm
#         response = response_chain.invoke({
#             "sku": product_details.sku,
#             "product_name": product_details.product_name,
#             "product_category": product_details.product_category,
#             "buyer_category": product_details.buyer_category
#         })

#         print(f"\n\nResponse: {response.content}")

#         return {
#             **state,
#             "product_name": product_details.product_name,
#             "product_category": product_details.product_category,
#             "buyer_category": product_details.buyer_category,
#             "current_node": END,
#             "conversation_history": state["conversation_history"] + [
#                 AIMessage(content=response.content)
#             ]
#         }
    
#     except Exception as e:
#         print("\nException in lookup_product_details:", repr(e))
#         # If the SKU cannot be found or something else goes wrong
#         not_found_prompt = ChatPromptTemplate.from_template(
#             """You are an audience building assistant for retail media.
            
#             The user asked about SKU {sku}, but it could not be found in our database.
            
#             Politely inform them that you couldn't find this SKU and ask if they'd like to try a different product SKU.

#             Respond as the assistant.
#             """
#         )

#         not_found_chain = not_found_prompt | llm
#         not_found_response = not_found_chain.invoke({"sku": product_sku})
        
#         return {
#             **state,
#             "conversation_history": state["conversation_history"] + [
#                 AIMessage(content=not_found_response.content)
#             ],
#             # If not found, route them back to identify_product
#             "current_node": END
#         }