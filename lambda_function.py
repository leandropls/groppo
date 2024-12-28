from os import environ
from typing import Annotated

from leandropls.easylambda import post
from leandropls.easylambda.body import Body
from leandropls.easylambda.errors import HttpBadRequest, HttpInternalServerError
from openai import OpenAI
from openai.types.beta.threads import RefusalContentBlock, TextContentBlock
from pydantic import BaseModel

# Get configuration from environment variables
api_key = environ["OPENAI_API_KEY"]
organization_id = environ["ORGANIZATION_ID"]
assistant_id = environ["ASSISTANT_ID"]
project_id = environ["PROJECT_ID"]

# Initialize OpenAI client
client = OpenAI(
    api_key=api_key,
    organization=organization_id,
    project=project_id,
)


# Send a message to a thread
class MessageRequest(BaseModel):
    message: str
    thread_id: str | None = None


class MessageResponse(BaseModel):
    response: str
    thread_id: str


@post("/messages")
def lambda_handler(request: Annotated[MessageRequest, Body]) -> MessageResponse:
    """Send a message to a thread and get the response."""
    # Create a new thread if one wasn't provided
    if request.thread_id is None:
        thread = client.beta.threads.create()
        thread_id = thread.id
    else:
        thread_id = request.thread_id

    # Add the user's message to the thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=request.message,
    )

    # Run the assistant
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

    # Check if the run failed
    if (incomplete_details := run.incomplete_details) is not None:
        match incomplete_details.reason:
            case "content_filter":
                raise HttpBadRequest("Message was blocked by the content filter")
            case _:
                raise HttpInternalServerError(
                    f"Message failed to run: {incomplete_details.reason}"
                )

    # Get the messages from the run
    messages_response = client.beta.threads.messages.list(
        thread_id=thread_id,
        run_id=run.id,
    )
    messages = []
    for item in messages_response:
        for content in item.content:
            match content:
                case TextContentBlock():
                    messages.append(content.text.value)
                case RefusalContentBlock():
                    messages.append(content.refusal)
                case _:
                    continue

    # Return the assistant's response
    if not messages:
        print(messages_response)
        return MessageResponse(
            response="I'm sorry, I don't have a response for that.",
            thread_id=thread_id,
        )

    return MessageResponse(
        response="\n\n".join(m.strip() for m in messages),
        thread_id=thread_id,
    )
