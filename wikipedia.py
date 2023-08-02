import asyncio
import os
from copy import copy
from pprint import pprint
from typing import List, Dict
from uuid import uuid4

import openai
import requests
import tiktoken
from bs4 import BeautifulSoup, Comment
from doctran import Doctran, ExtractProperty
from dotenv import load_dotenv, find_dotenv
from evaluate import load
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


def num_tokens_from_string(string: str, encoding_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError:
        encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def extract_content_from_wikipedia_url(url: str) -> Dict[str, str]:
    """Extracts the content from a Wikipedia URL into a dictionary. The keys are the header
    names + indicator of header level e.g. 'h2 Definitions'. The values are the content
    underneath each header tags.

    Only extracts content the user will see. Ignores hidden content and Contents list.

    Parameters
    ----------
    url : str
        The URL of the Wikipedia page to extract content from.

    Returns
    -------
    article_dict : dict
        A dictionary of the content extracted from the Wikipedia page.
    """
    if not "wikipedia" in url:
        raise ValueError("URL is not a wikipedia URL. Received: " + url)

    r = requests.get(url)
    html_content = r.text

    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove unwanted tags: script, style, [document], head, title
    for element in soup(["script", "style", "head", "title", "[document]"]):
        element.decompose()

    # Also remove HTML comments
    for element in soup.find_all(string=lambda text: isinstance(text, Comment)):
        element.extract()

    # Define the header tags to find
    tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
    found_tags = soup.find_all(tags)

    # Extract tags and associated content into a dict
    article_dict = {}
    for tag in found_tags:
        content = []
        next_tag = tag.find_next()

        # Look for next tags until the next header tag
        while next_tag and next_tag.name not in tags:
            # Reference section can contain both p and li tags
            if "reference" in str(next_tag).lower() and next_tag.name in ["p", "li"]:
                content.append(next_tag.get_text(strip=False))
            elif next_tag.name == "p":
                content.append(next_tag.get_text(strip=False))
            next_tag = next_tag.find_next()

        key = f"{tag.name} {tag.get_text(strip=True)}"
        article_dict[key] = " ".join(content)

    for key in list(
        article_dict.keys()
    ):  # Using list() to avoid changing the dictionary size during iteration
        if key.endswith("[edit]"):
            new_key = key.rsplit("[edit]", 1)[0]
            article_dict[new_key] = article_dict.pop(key)

    del article_dict["h2 Contents"]
    num_sections = len(article_dict.keys())

    logger.info(
        f"Sucessfully downloaded content from Wikipedia page. Extracted {num_sections} sections."
    )

    return article_dict


async def extract_title(string: str) -> str:
    """Extract a title from `string` that is max 7 words long."""
    doctran = Doctran(
        openai_api_key=os.getenv("OPENAI_API_KEY"), openai_model="gpt-3.5-turbo"
    )
    document = doctran.parse(content=string)
    properties = ExtractProperty(
        name="title",
        description="The title of the document (max 7 words).",
        type="string",
        required=True,
    )
    document = await document.extract(properties=[properties]).execute()
    return document.transformed_content


async def divide_sections_if_too_large_in_plan_and_sectionscontent(
    article_dict: Dict[str, str], max_section_length: int = 512
) -> Dict[str, str]:
    """This function takes an existing dictionary containing the plan and sections
    content (from above functions), checks if any section is too large (i.e., more
    than 512 tokens), divides such sections into smaller sections, generates a new
    title, and returns the updated dictionary
    """
    logger.info("Dividing sections if too large in plan and section content.")
    final_dict: Dict = {}
    start_dict = copy(article_dict)

    def is_reference_section(heading: str):
        """Returns True if heading is a reference section."""
        heading = heading.lower()
        result = (
            "reference" in heading
            or "further reading" in heading
            or "see also" in heading
        )
        return result

    for heading, content in start_dict.items():
        num_tokens = num_tokens_from_string(content)
        # Each section must contain something, otherwise the embedding models fail
        if num_tokens == 0:
            final_dict[heading] = " "
        # If the section is small enough, add it to the final dict
        elif num_tokens <= max_section_length:
            final_dict[heading] = content
        # If section is too big, split into smaller sections, extract title, and add to final dict
        else:
            # Split
            char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_section_length,
                chunk_overlap=0,
                # ' ' separator means sometimes sentences will be cut in two to ensure
                # the chunk size is not exceeded
                separators=["\n\n", "\n", " "],
                length_function=num_tokens_from_string,
            )
            splits: List[str] = char_splitter.split_text(content)
            # Keep heading the same but add numbers to sections e.g. 'h2 Reference' -> 'h2 Reference 1'
            if is_reference_section(heading):
                for i, split in enumerate(splits, start=1):
                    new_heading = f"{heading} {i}"
                    final_dict[new_heading] = split
                    logger.info(f"Added {new_heading} split original heading {heading}")
            # Split heading into more of the same level and add new title
            else:
                for split in splits:
                    # Headings are of the form h1, h2, h3 etc. we split it into more of the same level
                    heading_level = int(heading[1])
                    title = await extract_title(split)
                    new_heading = f"h{heading_level} {title}"
                    final_dict[new_heading] = split
                    logger.info(f"Added {new_heading} split original heading {heading}")

    n_keys_start = len(start_dict.keys())
    n_keys_final = len(final_dict.keys())
    logger.info(
        f"\n\tFinished dividing sections if too large in plan and section content."
        f"\n\tStarted with {n_keys_start} sections and got {n_keys_final} final sections."
        f"\n\tThat's a {n_keys_final / n_keys_start:.2f}x increase in sections"
    )
    return final_dict


def create_section_json(
    heading: str, content: str, id=1
) -> Dict[str, str | list[float]]:
    """Given a heading and content, returns a dictionary with the heading, content,
    and embeddings of the heading and content."""
    # Normalized by default
    embed_ada = OpenAIEmbeddings(
        model="text-embedding-ada-002",
    )

    embed_e5 = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2", encode_kwargs={"normalize_embeddings": True}
    )

    section_json = {
        "section_id": id,
        "section": heading,
        "content": content,
        "section_embedding_1": embed_ada.embed_query(heading),
        "section_embedding_2": embed_e5.embed_query("query: " + heading),
        "content_embedding_1": embed_ada.embed_query(content),
        "content_embedding_2": embed_e5.embed_query("query: " + content),
    }

    logger.info(f"section_id {id} - created section json + embeddings for {heading}")

    return section_json


def create_plan_json(article_dict: Dict) -> Dict:
    """Given a dictionary of the article content, returns a dictionary with the title,
    abstract, plan and associated embeddings.
    """
    logger.info("Creating plan json")
    headings = list(article_dict.keys())
    content = list(article_dict.values())

    title = headings[0][3:]
    abstract = content[0]
    plan = [
        create_section_json(heading, content, id=i)
        for i, (heading, content) in enumerate(zip(headings[1:], content[1:]), start=1)
    ]
    # Normalized by default
    embed_ada = OpenAIEmbeddings(
        model="text-embedding-ada-002",
    )

    embed_e5 = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2", encode_kwargs={"normalize_embeddings": True}
    )

    try:
        plan_json = {
            "id": str(uuid4()),
            "title": title,
            "abstract": abstract,
            "title_embedding_1": embed_ada.embed_query(title),
            "title_embedding_2": embed_e5.embed_query("query: " + title),
            "abstract_embedding_1": embed_ada.embed_query(abstract),
            "abstract_embedding_2": embed_e5.embed_query("query: " + abstract),
            "plan": plan,
            "plan_embedding_1": "???",
            "plan_embedding_2": "???",
            "embedding1_model": "text-embedding-ada-002",
            "embedding2_model": "e5-base-v2",
            "success": True,
            "error": None,
        }
    except Exception as e:
        plan_json = {
            "id": str(uuid4()),
            "title": title,
            "abstract": abstract,
            "title_embedding_1": "???",
            "title_embedding_2": "???",
            "abstract_embedding_1": "???",
            "abstract_embedding_2": "???",
            "plan": plan,
            "plan_embedding_1": "???",
            "plan_embedding_2": "???",
            "embedding1_model": "text-embedding-ada-002",
            "embedding2_model": "e5-base-v2",
            "success": False,
            "error": str(e),
        }

    logger.info("Finished creating plan json")
    return plan_json


def compare_plans(plan_1: Dict, plan_2: Dict) -> Dict:
    """Given two plans, returns a dictionary with the cosine similarity of the plans
    and the similarity scores of the plans (comparing both embeddings and text).
    """
    # TODO - how to handle the full content of each plan? It's way too big.
    plan_1_content: str = ...
    plan_2_content: str = ...

    # Mauve & Rouge have to compare the actual text strings themselves, not embeddings
    mauve = load("mauve")
    mauve_results = mauve.compute(
        predictions=[plan_1_content], references=[plan_2_content]
    )
    mauve_similarity = mauve_results.mauve

    rouge = load("rouge")
    results = rouge.compute(
        predictions=[plan_1_content],
        references=[plan_2_content],
        rouge_types=["rougeL"],
    )
    rouge_L_similarity = results["rougeL"]

    cosine_1 = cosine_similarity(
        [plan_1["plan_embedding_1"]], [plan_2["plan_embedding_1"]]
    )[0][0]
    cosine_2 = cosine_similarity(
        [plan_1["plan_embedding_2"]], [plan_2["plan_embedding_2"]]
    )[0][0]

    comparison_dict = {
        "document_id": str(uuid4()),
        "prediction_id": str(uuid4()),
        "plan_similarity": {
            "embedding1_cosine_similarity": cosine_1,
            "embedding2_cosine_similarity": cosine_2,
            "mauve_similarity": mauve_similarity,
            "rougeL_similarity": rouge_L_similarity,
        },
    }
    return comparison_dict


# TODO
def compare_content(plan_1: Dict, plan_2: Dict) -> Dict:
    pass


async def main(url: str) -> Dict:
    """Given a Wikipedia URL, returns a dictionary with the title, abstract, plan and
    associated embeddings.
    """
    article_dict = extract_content_from_wikipedia_url(url)
    article_dict = await divide_sections_if_too_large_in_plan_and_sectionscontent(
        article_dict
    )
    plan_json = create_plan_json(article_dict)
    return plan_json


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Dual-phase_evolution"
    url = "https://en.wikipedia.org/wiki/Self-driving_car"
    plan_json = asyncio.run(main(url))
    pprint(plan_json, sort_dicts=False)
