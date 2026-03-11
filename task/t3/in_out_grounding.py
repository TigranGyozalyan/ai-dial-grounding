import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from sqlalchemy.testing.suite.test_reflection import metadata

from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

CONTEXT_PROMPT = """
You are an LLM responsible for doing search based on hobbies. Your goal is to extract user hobbies from the context and group
users by their ids.

## Flow:
Step 1: User will ask to search users by their hobbies etc.
Step 2: Will be performed search in the Vector store to find most relevant users.
Step 3: You will be provided with CONTEXT (most relevant users, there will be user ID and information about user), and 
        with USER QUESTION.
Step 4: You group by hobby users that have such hobby and return response according to Response Format

## Response Format:
{format_instructions}
"""

class GroupingResult(BaseModel):
    hobby: str = Field(description="Hobby", examples=["football" "painting", "horsing", "photography", "bird watching"])
    user_ids: list[int] = Field(description="List of user IDs that have hobby requested by user.")


class GroupingResults(BaseModel):
    grouping_results: list[GroupingResult] = Field(description="List matching search results.")

def format_user_document(user: dict[str, Any]) -> str:
    return f"User:\n name: {user['id']}\n about_me: {user['about_me']}"

def as_batches(users: list[Any], batch_size: int = 100) -> list[list[Any]]:
    batches = []
    idx = 0
    for user in users:
        batch_idx = int(idx / batch_size)
        if batch_idx not in range(len(batches)):
            batches.append([])
        batches[batch_idx].append(user)
        idx += 1
    return batches


class InputGrounding:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore: Chroma = None
        self.user_client = UserClient()

    async def __aenter__(self):
        print("🔎 Loading all users...")
        users = self.user_client.get_all_users()
        docs = [Document(id=user['id'], page_content=format_user_document(user)) for user in users]

        self.vectorstore = await self._create_vectorstore_with_batching(documents=docs)
        print("✅ Vectorstore is ready.")
        return self

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100) -> Chroma:
        final_store = Chroma(
            collection_name="user_info",
            embedding_function=self.embeddings,
            persist_directory="./chroma_langchain_db"
        )

        doc_batches = as_batches(documents, batch_size)
        tasks = []

        for doc_batch in doc_batches:
            tasks.append(final_store.aadd_documents(documents=doc_batch, embedding=self.embeddings))

        await asyncio.gather(*tasks)
        return final_store

    async def get_context(self, user_query: str, k: int = 10, score: float = 0.4) -> str:
        print("syncing context...")
        await self.reconcile_context()
        print("context is up-to-date!")
        result = self.vectorstore.similarity_search_with_relevance_scores(user_query, k=k, score_threshold=score)
        context_parts = []
        for (doc, score) in result:
            context_parts.append(doc.page_content)
            print(f"content: {doc.page_content}, score: {score}")

        return "\n\n".join(context_parts)

    def generate_grounded_context(self, vector_context: str) -> GroupingResults:
        parser = PydanticOutputParser(pydantic_object=GroupingResults)
        messages = [
            SystemMessagePromptTemplate.from_template(template=CONTEXT_PROMPT),
            HumanMessage(content=vector_context)
        ]

        prompt = ChatPromptTemplate(messages=messages).partial(format_instructions=parser.get_format_instructions())

        result: GroupingResults = (prompt | self.llm_client | parser).invoke({})
        return result

    async def reconcile_context(self):
        users = self.user_client.get_all_users()
        vectorstore_data = self.vectorstore.get()
        vectorstore_ids_set = set(str(user_id) for user_id in vectorstore_data.get("ids", []))

        users_dict = {str(user.get('id')): user for user in users}
        users_ids_set = set(users_dict.keys())

        new_user_ids = users_ids_set - vectorstore_ids_set
        ids_to_delete = vectorstore_ids_set - users_ids_set

        new_documents = [
            Document(id=user_id, page_content=format_user_document(users_dict[user_id]))
            for user_id in new_user_ids
        ]

        if ids_to_delete:
            self.vectorstore.delete(list(ids_to_delete))

        if new_documents:
            if len(new_documents) > 50:
                batches = [new_documents[i:i + 50] for i in range(0, len(new_documents), 50)]
                await asyncio.gather(*[self.vectorstore.aadd_documents(batch) for batch in batches])
            else:
                await self.vectorstore.aadd_documents(new_documents)


    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class OutputGrounding:
    def __init__(self):
        self.user_client = UserClient()

    async def ground_response(self, grouping_results: GroupingResults):
        for grouping_result in grouping_results.grouping_results:
            print(f"Hobby: {grouping_result.hobby}\n")
            print(f"Users:\n {await self._find_users(grouping_result.user_ids)}\n")
            print("----------\n")


    async def _find_users(self, ids: list[int]) -> list[dict[str, Any]]:
        async def safe_get_user(user_id: int) -> Optional[dict[str, Any]]:
            try:
                return await self.user_client.get_user(user_id)
            except Exception as e:
                if "404" in str(e):
                    print(f"User with ID {user_id} is absent (404)")
                    return None
                raise  # Re-raise non-404 errors

        tasks = [safe_get_user(user_id) for user_id in ids]
        users_results = await asyncio.gather(*tasks)

        return [user for user in users_results if user is not None]


async def main() -> None:
    llm_client = AzureChatOpenAI(
        temperature=0.0,
        api_key=SecretStr(API_KEY),
        azure_endpoint=DIAL_URL,
        azure_deployment='gpt-4o',
        api_version='',
    )
    embeddings = AzureOpenAIEmbeddings(
        api_key=SecretStr(API_KEY),
        azure_endpoint=DIAL_URL,
        azure_deployment='text-embedding-3-small-1',
        dimensions=384
    )
    async with InputGrounding(llm_client=llm_client, embeddings=embeddings) as rag:
        output_grounder = OutputGrounding()
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find people who love to watch stars and night sky")
        print(" - I need people to go to fishing together")
        while True:
            user_question = input("What is your question? ")
            context = await rag.get_context(user_query=user_question)
            groupings = rag.generate_grounded_context(context)
            await output_grounder.ground_response(groupings)


if __name__ == "__main__":
    asyncio.run(main())