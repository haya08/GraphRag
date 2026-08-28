import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase

from neo4j_graphrag.llm import LLMBase, LLMResponse
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG


load_dotenv()


# ============================================================
# 1. Neo4j Configuration
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

if not NEO4J_URI:
    raise ValueError("NEO4J_URI is not configured.")

if not NEO4J_DATABASE:
    raise ValueError("NEO4J_DATABASE is not configured.")


# ============================================================
# 2. Connect to Existing Neo4j Knowledge Graph
# ============================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
)

driver.verify_connectivity()

print("Connected to Neo4j!")


# ============================================================
# 3. Existing BSKO Graph Schema
# ============================================================

SCHEMA = """
Node labels and properties:

Brand:
    name: STRING

Product:
    name: STRING

Feature:
    name: STRING

Person:
    name: STRING

Campaign:
    name: STRING


Relationships:

(:Brand)-[:HAS_PRODUCT]->(:Product)

(:Brand)-[:HAS_FEATURE]->(:Feature)

(:Campaign)-[:LED_BY]->(:Person)
"""


# ============================================================
# 4. Gemini
# ============================================================

gemini = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)


# ============================================================
# 5. Gemini -> Neo4j GraphRAG Adapter
# ============================================================

def extract_text(content) -> str:

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    return str(content)


class GeminiLLM(LLMBase):

    def __init__(self):

        super().__init__(
            model_name="gemini-3.6-flash"
        )

    def invoke(
        self,
        input,
        message_history=None,
        system_instruction=None,
        **kwargs
    ):

        prompt = input

        if system_instruction:

            prompt = f"""
System instruction:

{system_instruction}

User:

{input}
"""

        response = gemini.invoke(prompt)

        return LLMResponse(
            content=extract_text(
                response.content
            )
        )

    async def ainvoke(
        self,
        input,
        message_history=None,
        system_instruction=None,
        **kwargs
    ):

        prompt = input

        if system_instruction:

            prompt = f"""
System instruction:

{system_instruction}

User:

{input}
"""

        response = await gemini.ainvoke(prompt)

        return LLMResponse(
            content=extract_text(
                response.content
            )
        )


gemini_graphrag = GeminiLLM()


# ============================================================
# 6. Text2Cypher Retriever
# ============================================================

retriever = Text2CypherRetriever(
    driver=driver,
    llm=gemini_graphrag,
    neo4j_schema=SCHEMA,
    neo4j_database=NEO4J_DATABASE
)


# ============================================================
# 7. GraphRAG Pipeline
# ============================================================

rag = GraphRAG(
    retriever=retriever,
    llm=gemini_graphrag
)


# ============================================================
# 8. Ask Question
# ============================================================

if __name__ == "__main__":

    q = """
i want you to return the names of the products that brand "Nike" provides,
and the features that each of these products has, not the features that "Nike" provides.
"""

    result = retriever.search(q)


    # ========================================================
    # Final Answer
    # ========================================================

    print("\n" + "=" * 60)
    print("GENERATED CYPHER")
    print("=" * 60)
    print(result.metadata.get("cypher"))

    print("\n" + "=" * 60)
    print("NEO4J RESULT")
    print("=" * 60)

    for item in result.items:
        print(item.content)


    # ========================================================
    # Cleanup
    # ========================================================

    driver.close()