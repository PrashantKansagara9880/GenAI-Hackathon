import functools
import json
import socket

import groq
from google.genai import errors as genai_errors
from langgraph.errors import GraphRecursionError
from langchain_core.exceptions import OutputParserException


def safe_call(func):
    """
    Decorator that catches common exceptions raised by
    LLMs, tools, LangGraph, networking and JSON parsing.

    Instead of crashing the application, it returns
    a friendly error message.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        try:
            return func(*args, **kwargs)

        except GraphRecursionError:
            return (
                "⚠️ The agent exceeded its reasoning limit.\n"
                "Try asking a simpler question."
            )

        except groq.RateLimitError:
            return (
                "⏳ Groq rate limit reached.\n"
                "Please try after few moments."
            )

        except groq.AuthenticationError:
            return (
                "🔑 Invalid Groq API key."
            )

        except groq.PermissionDeniedError:
            return (
                "🚫 Permission denied."
            )

        except groq.NotFoundError:
            return (
                "❌ Requested model was not found."
            )

        except groq.APIConnectionError:
            return (
                "🌐 Could not connect to Groq.\n"
                "Check your internet connection."
            )

        except groq.InternalServerError:
            return (
                "⚙️ Groq server is temporarily unavailable."
            )

        except groq.BadRequestError as e:
            return (
                f"❌ Invalid request:\n{e}"
            )

        except json.JSONDecodeError:
            return (
                "⚠️ The model returned invalid JSON.\n"
                "Please try again."
            )
        

        except OutputParserException:
            return (
                "⚠️ Failed to parse the model output."
            )

        except TimeoutError:
            return (
                "⌛ Request timed out."
            )

        except socket.timeout:
            return (
                "⌛ Network timeout."
            )

        except ConnectionError:
            return (
                "🌐 Connection error."
            )

        except FileNotFoundError as e:
            return (
                f"📄 File not found:\n{e}"
            )

        except PermissionError:
            return (
                "🔒 Permission denied while accessing a file."
            )

        except ValueError as e:
            return (
                f"⚠️ Invalid input:\n{e}"
            )

        except KeyError as e:
            return (
                f"⚠️ Missing key: {e}"
            )

        except genai_errors.ClientError as e:
            status=getattr(e, "status_code", None)
            if status == 429:
                return (
            "⏳ Gemini is currently under high demand.\n"
            "Please try again in a few moments."
        )
            elif status == 404:
                return "❌ Gemini model not found."
            elif status == 401:
                return "🔑 Invalid Gemini API key."
            return f"❌ Gemini Client Error: {e}"
        
        except genai_errors.ServerError as e:
            return (
        "⚙️ Gemini servers are temporarily busy.\n"
        "Please try again shortly."
    )
        
        except Exception as e:
            return (
                f"❌ Unexpected error:\n{type(e).__name__}: {e}"
            )

    return wrapper