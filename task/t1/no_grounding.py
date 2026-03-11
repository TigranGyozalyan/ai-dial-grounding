import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient


BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }

token_tracker = TokenTracker()
azure_client = AzureChatOpenAI(azure_endpoint=DIAL_URL,api_key=API_KEY, azure_deployment='gpt-4o', api_version='')

def join_context(context: list[dict[str, Any]]) -> str:
    formatted = []

    for user in context:
        formatted.append(
            f"""
            User:
              name: {user['name']}
              surname: {user['surname']}
              about_me: {user['about_me']}
              gender: {user['gender']}
              email: {user['email']}
              company: {user['company']}
              salary: {user['salary']}
            """)
    return "\n\n".join(formatted)


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    response = await azure_client.ainvoke(messages)

    total_tokens = response.response_metadata['token_usage']['total_tokens']
    token_tracker.add_tokens(total_tokens)

    content = response.content
    print(f"response: {content}, tokens: {total_tokens}")
    return content

async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")
    user_client = UserClient()
    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")
        users = user_client.get_all_users()
        batches = as_batches(users)
        tasks = []
        for batch in batches:
            context = join_context(batch)
            tasks.append(generate_response(BATCH_SYSTEM_PROMPT, augmented_user_prompt(context, user_question)))

        print("\n--- Looking for matches ---")
        results = await asyncio.gather(*tasks)
        matches = []
        for result in results:
            if not result == 'NO_MATCHES_FOUND':
                matches.append(result)
            else:
                print("No matches found")

        matches_as_str = "\n\n".join(matches)
        await generate_response(FINAL_SYSTEM_PROMPT, augmented_user_prompt(matches_as_str, user_question))
        print(f"total token summary: {token_tracker.get_summary()}")

def augmented_user_prompt(context: str, query: str) -> str:
    return USER_PROMPT.replace("{context}", context).replace("{query}", query)

def as_batches(users: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches = []
    idx = 0
    for user in users:
        batch_idx = int(idx / 100)
        if batch_idx not in range(len(batches)):
            batches.append([])
        batches[batch_idx].append(user)
        idx += 1
    return batches

if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation