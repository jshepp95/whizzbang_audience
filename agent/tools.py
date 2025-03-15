import sqlite3

from langchain.tools import BaseTool
from typing import Type, ClassVar
from pydantic import BaseModel, Field
from schema import ProductDetails, ProductSearchResults

from collections import defaultdict

DB_PATH = "/home/azureuser/projects/whizzbang_audience/db/db.db"

class SKULookupInput(BaseModel):
    sku: str = Field(..., description="The product SKU to lookup")

class ProductLookupInput(BaseModel):
    sku: str = Field(..., description="The product name to lookup")

class SKULookupTool(BaseTool):
    name: ClassVar[str] = "product_database_lookup"
    description: ClassVar[str] = "Use this tool to look up a product in the database by its SKU"
    args_schema: ClassVar[Type[BaseModel]] = SKULookupInput

    def _run(self, sku: str) -> ProductDetails:
        """ Query the database for product details """
        db_path = DB_PATH

        try:
            print(f"Querying database for SKU: {sku}")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            query = """
            SELECT skuId, skuName, catLevel4Name, catLevel5Name
            FROM DIM_ITEMS
            WHERE skuId = ?
            """
            cursor.execute(query, (sku,))
            result = cursor.fetchone()
            conn.close()

            print(result)

            if result:
                return ProductDetails(
                    sku=result[0],
                    product_name=result[1],
                    buyer_category=result[2],
                    product_category=result[3]
                )
            else:
                raise ValueError(f"Product with SKU {sku} not found")
            
        except sqlite3.Error as e:
            raise ValueError(f"DB Error: {e}")
        

class ProductLookupTool(BaseTool):
    name: ClassVar[str] = "product_database_lookup"
    description: ClassVar[str] = "Use this tool to look up a product in the database by its name"
    args_schema: ClassVar[Type[BaseModel]] = ProductLookupInput

    def _run(self, name: str) -> ProductSearchResults:
        """ Query the database for product details and group by categories """
        db_path = DB_PATH

        try:
            print(f"Querying database for name: {name}")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            query = """
            SELECT 
                skuId,
                skuName,
                catLevel4Name,
                catLevel5Name
            FROM 
                DIM_ITEMS
            WHERE 
                skuName LIKE ? || '%' OR
                skuName LIKE '%' || ? OR
                skuName LIKE '%' || ? || '%'
            ORDER BY 
                CASE
                    WHEN skuName = ? THEN 10
                    WHEN skuName LIKE ? || '%' THEN 8
                    WHEN skuName LIKE '%' || ? || '%' THEN 6
                    ELSE 1
                END DESC
            LIMIT 10;
            """
            
            params = (name, name, name, name, name, name)
            cursor.execute(query, params)
            
            results = cursor.fetchall()
            conn.close()

            print(f"Found {len(results)} results")
            
            if results:
                # Build a dictionary to group products by categories
                by_buyer_category = defaultdict(list)
                by_product_category = defaultdict(list)
                
                # Also track unique categories
                unique_buyer_categories = set()
                unique_product_categories = set()
                
                # Process all results
                all_products = []
                for row in results:
                    sku_id = row[0]
                    product_name = row[1]
                    buyer_category = row[2]
                    product_category = row[3]
                    
                    # Create product object
                    product = ProductDetails(
                        sku=sku_id,
                        product_name=product_name,
                        buyer_category=buyer_category,
                        product_category=product_category
                    )
                    
                    # Add to our collections
                    all_products.append(product)
                    by_buyer_category[buyer_category].append(product)
                    by_product_category[product_category].append(product)
                    
                    # Track unique categories
                    unique_buyer_categories.add(buyer_category)
                    unique_product_categories.add(product_category)
                
                # Build the response object
                response = ProductSearchResults(
                    query=name,
                    total_results=len(results),
                    unique_buyer_categories=list(unique_buyer_categories),
                    unique_product_categories=list(unique_product_categories),
                    by_buyer_category=dict(by_buyer_category),
                    by_product_category=dict(by_product_category),
                    all_products=all_products
                )
                
                print(f"\nFound products in {len(unique_buyer_categories)} buyer categories and {len(unique_product_categories)} product categories\n")
                
                return response
            else:
                raise ValueError(f"Product with name {name} not found")
            
        except sqlite3.Error as e:
            raise ValueError(f"DB Error: {e}")
        

            




