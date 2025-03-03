import sqlite3

from langchain.tools import BaseTool
from typing import Type, ClassVar
from pydantic import BaseModel, Field
from schema import ProductDetails

DB_PATH = "db/db.db"

class SKULookupInput(BaseModel):
    sku: str = Field(..., description="The product SKU to lookup")

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
            SELECT skuID, skuName, catLevel4, catLevel5,
            FROM ITEMS
            WHERE skuID = ?
            """
            cursor.execute(query, (sku,))
            result = cursor.fetchone()

            print(result)

            conn.close()

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
        

            




