import asyncio
import json
import os
import re
from collections import defaultdict
from copy import copy
from pathlib import Path
from pprint import pprint
from time import time
from typing import List, Dict, Any
from uuid import uuid4

import numpy as np
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
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


def _num_tokens_from_string(string: str, encoding_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError:
        encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncated_pprint(obj, N=5):
    """Pretty print an object, truncating lists and strings to N items/characters
    for easier viewing of plan_json objects"""

    def truncate(item, N):
        if isinstance(item, list) and N is not None:
            return item[:N] + (["..."] if len(item) > N else [])
        if isinstance(item, str) and N is not None:
            N = 125
            return item[:N] + ("..." if len(item) > N else "")
        return item

    def trunc_recursive(item, N):
        if isinstance(item, list):
            return [trunc_recursive(i, N) for i in truncate(item, N)]
        elif isinstance(item, dict):
            return {k: trunc_recursive(v, N) for k, v in item.items()}
        else:
            return truncate(item, N)

    truncated_obj = trunc_recursive(obj, N)
    pprint(truncated_obj, sort_dicts=False)


def load_arxiv_paper(path: str | Path) -> Dict[str, str]:
    """Load an arXiv paper from disk to a dict with keys "Title", "Abstract", "Content",
    and "References"."""
    doc = Path(path)
    # Extract all text from pdf
    text = extract_text(doc)

    title = doc.stem
    title = title.replace("_", " ")

    # The pattern searches for "abstract" followed by any content.
    # Then, it looks for one of the potential following sections:
    # "I. Introduction", "1. Introduction", or "Contents".
    # We use a positive lookahead (?=...) to assert that the introduction or contents
    # pattern exists, but we don't include it in the main match.
    pattern = r"abstract(.*?)(?=(i\. introduction|1\. introduction|contents))"

    # The re.DOTALL flag allows the . in the pattern to match newline characters,
    match = re.search(pattern, text.lower(), re.DOTALL)

    abstract = ""
    abstract_end = 0
    if match:
        abstract_end = match.end()
        abstract = match.group(1).strip()

    # Extract references section
    pattern = r"references\n"
    matches = [match for match in re.finditer(pattern, text.lower())]

    references = ""
    reference_start = len(text)
    if matches:
        final_match = matches[-1]
        reference_start = final_match.start()
        references = text[reference_start:]

    content = text[abstract_end:reference_start]

    article_dict = {
        "Title": title,
        "Abstract": abstract,
        "Content": content,
        "References": references,
    }
    return article_dict


def split_patents_into_individual_files(patents_file: str | Path) -> None:
    """Read in a file containing many patents. Split each patent into its own file, keeping
    only the english parts, and write to disk."""
    # Read in file
    with open(patents_file, "r") as f:
        lines = f.readlines()
    # Get all eng sections
    lines_en = [line for line in lines if "\ten\t" in line]
    # Split each on TITLE and write to its own file with TITLE as filename
    os.makedirs("data/patents", exist_ok=True)
    title = "no title found"
    # Create dict of patents
    patents: defaultdict = defaultdict(str)
    for i, x in enumerate(lines_en):
        if "\tTITLE\t" in x:
            title = x.split("\t")[-1].strip()
        patents[title] += x

    # Write each patent to its own file
    for title, content in patents.items():
        filename_friendly_title = "".join(i for i in title if i not in "\/:*?<>|")
        with open(f"data/patents/{filename_friendly_title}.txt", "w") as f:
            f.write(content)
            logger.info(f"Wrote file: {filename_friendly_title}.txt")


def load_patent_file(patent_file: str | Path) -> Dict[str, str]:
    """Read in a patent file and return a dict with keys as section titles and values the content.

    Parameters
    ----------
    patent_file : str
        Path to the patent file.

    Returns
    -------
    patent_dict : dict
        Dict with keys as section titles and values the content. Keys are ['title',
        'descr', 'claim_1', 'claim_2', ..., 'claim_n', 'pdfep']. Not all patents
        will have all keys. All will have 'title' at a minimum.
    """
    logger.info(f"Loading patent file: {patent_file}")
    # Read file
    with open(patent_file, "r") as f:
        lines: list = f.readlines()

    # Get all english sections
    lines_en: list = [line for line in lines if "\ten\t" in line]

    # Convert into dict with keys as section titles and values the content
    patent_dict = {}
    total_claims = 1
    for x in lines_en:
        if "\tTITLE\t" in x:
            patent_dict["title"] = x
        elif "\tDESCR\t" in x:
            patent_dict["descr"] = x
        elif "\tCLAIM\t" in x:
            # Some patents have multiple claims, so we need to number them
            patent_dict[f"claim_{total_claims}"] = x
            total_claims += 1
        elif "\tPDFEP" in x:
            patent_dict["pdfep"] = x
        else:
            raise ValueError(
                f"Expected sections in [TITLE, DESCR, CLAIM, PDFEP]. Received: {x}"
            )

    logger.info(
        f"Extracted {len(patent_dict)} sections from patent file named: "
        f"{list(patent_dict.keys())}"
    )

    return patent_dict


def load_wikipedia_url(url: str) -> Dict[str, str]:
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
        f"Successfully downloaded content from Wikipedia page {url}. "
        f"Extracted {num_sections} sections."
    )

    return article_dict


async def _extract_title(string: str) -> str:
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
    try:
        document = await document.extract(properties=[properties]).execute()
        return document.transformed_content
    except Exception as e:
        logger.error(f"Error extracting title from string: {e}")
        return "None"


async def divide_sections_if_too_large(
    article_dict: Dict[str, str],
    max_section_length: int = 512,
    doc_type: str = "patent",
) -> Dict[str, str]:
    """This function takes an existing dictionary containing the section heaadings and
    content (from above functions), checks if any section is too large (i.e., more
    than 512 tokens), divides such sections into smaller sections, generates a new
    title, and returns the updated dictionary
    """
    if doc_type not in ["patent", "wikipedia", "arxiv"]:
        raise ValueError(
            f"doc_type must be one of 'patent', 'wikipedia', or 'arxiv'. Got {doc_type}."
        )
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
        num_tokens = _num_tokens_from_string(content)
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
                length_function=_num_tokens_from_string,
            )
            splits: List[str] = char_splitter.split_text(content)
            # Keep heading the same but add numbers to sections e.g. 'h2 Reference' -> 'h2 Reference 1'
            # TODO - add a continue statement here?
            if doc_type in ["wikipedia", "arxiv"] and is_reference_section(heading):
                for i, split in enumerate(splits, start=1):
                    new_heading = f"{heading} {i}"
                    final_dict[new_heading] = split
                    logger.info(
                        f"Added '{new_heading}' split original heading '{heading}'"
                    )
            else:
                # Create new titles for each split
                for split in splits:
                    # Headings are of the form h1, h2, h3 etc. we split it into more of the same level
                    if doc_type == "wikipedia":
                        heading_level = int(heading[1])
                        title = await _extract_title(split)
                        new_heading = f"h{heading_level} {title}"
                    # Heading levels aren't important for other doc_types
                    else:
                        new_heading = await _extract_title(split)
                    final_dict[new_heading] = split
                    logger.info(
                        f"Added '{new_heading}' split original heading '{heading}'"
                    )

    n_keys_start = len(start_dict.keys())
    n_keys_final = len(final_dict.keys())
    logger.info(
        f"\n\tFinished dividing sections if too large in plan and section content."
        f"\n\tStarted with {n_keys_start} sections and got {n_keys_final} final sections."
        f"\n\tThat's a {n_keys_final / n_keys_start:.2f}x increase in sections"
    )
    return final_dict


def _gen_embed_section_content(
    heading: str, content: str, id: int = 1, total_sections: int = 1
) -> Dict[str, str | list[float]]:
    """Given a heading and content, returns a dictionary with the heading, content,
    and embeddings of the heading and content.

    Parameters
    ----------
    heading : str
        The heading of the section.
    content : str
        The content of the section.
    id : int, optional
        The id of the section, by default 1
    total_sections : int, optional
        The total number of sections, by default 1
    """
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

    logger.info(
        f"{id}/{total_sections} - created section + content embeddings for {heading}"
    )

    return section_json


def _gen_embed_plan(plan: List[dict], i: int) -> List[float]:
    """Calculate plan embedding by averaging the section embeddings and content embeddings
    sequentially.

    Parameters
    ----------
    plan : List[dict]
        List of dictionaries, each containing the section and content embeddings.
    i : int
        The index of the embedding to use.
    """
    try:
        s1_mean = np.mean([x[f"section_embedding_{i}"] for x in plan], axis=0)
        c1_mean = np.mean([x[f"content_embedding_{i}"] for x in plan], axis=0)
    except KeyError:
        raise KeyError(
            f"Could not find section_embedding_{i} or content_embedding_{i} in "
            f"every element in plan. Please check that every element has both of these "
            f"keys."
        )
    total_mean = np.mean([c1_mean, s1_mean], axis=0)
    total_mean = list(total_mean)
    logger.info(f"Created plan embedding {i}")
    return total_mean


def generate_embeddings_plan_and_section_content(
    article_dict: Dict, doc_type: str = "patent"
) -> Dict:
    """Given a dictionary of the article content, returns a dictionary with the title,
    abstract, plan and associated embeddings.
    """
    doc_type_error_msg = (
        f"doc_type must be one of 'patent', 'wikipedia', or 'arxiv'. "
        f"Received {doc_type}"
    )
    if doc_type not in ["patent", "wikipedia", "arxiv"]:
        raise ValueError(doc_type_error_msg)
    logger.info("Creating plan json")
    headings = list(article_dict.keys())
    content = list(article_dict.values())

    if doc_type == "wikipedia":
        # First key/value pair in wikipedia dict is {title: abstract}. So we can take
        # the first heading and first content

        # Wikipedia titles are of form 'h1 Example' so we remove the 'h1 '
        title = headings[0][3:]
        abstract = content[0]
        total_sections = len(headings) - 1
        start_index = 1
    elif doc_type == "patent":
        # First key/value pairs in patent dict are {'title': title, 'abstract': abstract}
        # so we take the first two elements of content

        # Titles are separated by tabs, the last element is the actual title
        title = content[0].split("\t")[-1].strip()
        # Remove illegal characters from title (it's used as a filename)
        title = "".join(i for i in title if i not in "\/:*?<>|")
        try:
            abstract = content[1]
        except IndexError:
            abstract = "no abstract"
        total_sections = len(headings) - 2
        start_index = 2
    elif doc_type == "arxiv":
        # The first key/value pairs in arxiv dicts are {'Title': title, 'Abstract': abstract}
        # so we take the first two elements of content
        title = content[0]
        try:
            abstract = content[1]
        except IndexError:
            abstract = "no abstract"
        total_sections = len(headings) - 2
        start_index = 2
    else:
        raise ValueError(doc_type_error_msg)

    logger.info("Title: " + title)
    logger.info("Abstract: " + abstract)

    plan = [
        _gen_embed_section_content(
            heading, content, id=i, total_sections=total_sections
        )
        for i, (heading, content) in enumerate(
            zip(headings[start_index:], content[start_index:]), start=1
        )
    ]
    plan_embed_1 = _gen_embed_plan(plan, 1)
    plan_embed_2 = _gen_embed_plan(plan, 2)
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
            "plan_embedding_1": plan_embed_1,
            "plan_embedding_2": plan_embed_2,
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
            "title_embedding_1": None,
            "title_embedding_2": None,
            "abstract_embedding_1": None,
            "abstract_embedding_2": None,
            "plan": plan,
            "plan_embedding_1": None,
            "plan_embedding_2": None,
            "embedding1_model": "text-embedding-ada-002",
            "embedding2_model": "e5-base-v2",
            "success": False,
            "error": str(e),
        }

    logger.info("Finished creating plan json")
    return plan_json


async def get_embeddings(
    input: str | List[str], model: str = "text-embedding-ada-002"
) -> List[float] | List[List[float]]:
    """This function takes one string or a list of strings and a model name,
    generates an embedding for each string in the list using the specified model,
    and returns a list of embeddings, each represented as a list of floating point
    numbers.

    Parameters
    ----------
    input : str | List[str]
        The input string or list of strings to be embedded.
    model : str, optional ['text-embedding-ada-002', 'e5-base-v2']
        The name of the model to be used for embedding.
    """
    if model == "text-embedding-ada-002":
        embedder = OpenAIEmbeddings(model=model)
    elif model == "e5-base-v2":
        embedder = HuggingFaceEmbeddings(
            model_name=f"intfloat/{model}", encode_kwargs={"normalize_embeddings": True}
        )
    else:
        raise ValueError(
            f"Model name must be 'text-embedding-ada-002' or 'e5-base-v2'. Received {model}"
        )

    if isinstance(input, str):
        try:
            return await embedder.aembed_query(input)
        except NotImplementedError as e:
            return embedder.embed_query(input)
    elif isinstance(input, list):
        try:
            return await embedder.aembed_documents(input)
        except NotImplementedError as e:
            return embedder.embed_documents(input)
    else:
        raise ValueError(
            f"Input must be a string or a list of strings. Received {type(input)}"
        )


def _compare_documents(
    document: str | Path | Dict[str, Any],
    prediction: str | Path | Dict[str, Any],
    compare_on: str = "section",
) -> Dict[str, Any]:
    """Compare the 'compare_on' sections of document and prediction. Calculate MAUVE,
    and ROUGE-L scores on the actual text, and cosine similarity on the embeddings.

    Parameters
    ----------
    document : Dict[str, Any]
        Dictionary containing the grouth truth document. Must contain the keys
        'plan' and 'id'.
    prediction : Dict[str, Any]
        Dictionary containing the prediction to compare against. Must contain the keys
        'plan' and 'id'.
    compare_on : str, optional ['section', 'content']
        Whether to compare on the 'section' level i.e. the plan of the document, or
        the 'content' level.
    """
    if compare_on not in ["section", "content"]:
        raise ValueError(
            f"`compare_on` must be 'section' or 'content'. Received {compare_on}"
        )

    if isinstance(document, str) or isinstance(document, Path):
        with open(document, "r") as f:
            document = json.load(f)

    if isinstance(prediction, str) or isinstance(prediction, Path):
        with open(prediction, "r") as f:
            prediction = json.load(f)

    if not isinstance(document, dict) or not isinstance(prediction, dict):
        raise TypeError(
            "Both `document` and `prediction` must be dictionaries. Received "
            f"{type(document)} and {type(prediction)}"
        )

    if "plan" not in document or "plan" not in prediction:
        raise ValueError(
            f'Both `document` and `prediction` must contain the key "plan". At least '
            f"one of them does not."
        )

    start = time()
    doc1_name = f"ID: {document['id']} Title: {document['title']}"
    doc2_name = f"ID: {prediction['id']} Title: {prediction['title']}"
    logger.info(
        f"\n\tStarting to compare two documents on {compare_on}:"
        f"\n\t\t{doc1_name}"
        f"\n\t\t{doc2_name}"
    )

    mauve = load("mauve")
    rouge = load("rouge")

    section_results = []
    doc_plan: List[Dict[str, Any]] = document["plan"]
    predict_plan: List[Dict[str, Any]] = prediction["plan"]

    logger.info(
        f"\n\t{doc1_name} has {len(doc_plan)} sections."
        f"\n\t{doc2_name} has {len(predict_plan)} sections."
    )
    total_comparisons = min(len(doc_plan), len(predict_plan))
    # If plans have differnet lengths, just goes up to shortest
    for idx, (p_dict, d_dict) in enumerate(zip(predict_plan, doc_plan), start=1):
        # Compute MAUVE
        mauve_results = mauve.compute(
            predictions=[p_dict[compare_on]], references=[d_dict[compare_on]]
        )
        mauve_score = mauve_results.mauve
        # Compute ROUGE-L
        results = rouge.compute(
            predictions=[p_dict[compare_on]],
            references=[d_dict[compare_on]],
            rouge_types=["rougeL"],
        )
        rouge_score = results["rougeL"]
        # Compute cosine distance between both section embeddings
        cosine_1 = cosine_similarity(
            [p_dict[f"{compare_on}_embedding_1"]], [d_dict[f"{compare_on}_embedding_1"]]
        )[0][0]
        cosine_2 = cosine_similarity(
            [p_dict[f"{compare_on}_embedding_2"]], [d_dict[f"{compare_on}_embedding_2"]]
        )[0][0]
        # Combine results
        result = {
            "section_id": idx,
            "mauve_similarity": mauve_score,
            "rouge_L_similarity": rouge_score,
            "embedding1_cosine_similarity": cosine_1,
            "embedding2_cosine_similarity": cosine_2,
        }
        section_results.append(result)
        logger.info(f"{idx}/{total_comparisons} sections compared.")

    # Calcualte total scores
    mauve_total = np.mean([x["mauve_similarity"] for x in section_results])
    rouge_total = np.mean([x["rouge_L_similarity"] for x in section_results])
    cosine_1_total = np.mean(
        [x["embedding1_cosine_similarity"] for x in section_results]
    )
    cosine_2_total = np.mean(
        [x["embedding2_cosine_similarity"] for x in section_results]
    )

    total_results = {
        "mauve_similarity": mauve_total,
        "rouge_L_similarity": rouge_total,
        "embedding1_cosine_similarity": cosine_1_total,
        "embedding2_cosine_similarity": cosine_2_total,
    }

    if compare_on == "section":
        compare_on = "plan"

    output = {
        "document_id": document["id"],
        "prediction_id": prediction["id"],
        f"{compare_on}_total_similarity": total_results,
        f"{compare_on}_bysection_similarity": section_results,
    }

    end = time()
    seconds = end - start
    mins = seconds / 60
    logger.info(
        f"\n\tFinished comparing document {compare_on}s:"
        f"\n\t\tThat took: {mins:.2f} mins ({seconds:.0f} seconds)"
    )
    return output


def compare_documents_sections(
    document1: str | Path | Dict[str, Any],
    document2: str | Path | Dict[str, Any],
) -> Dict[str, Any]:
    """This function takes two documents, a comparison method, compares the section
    headings (also called plans) of the documents in order using the specified method,
    and returns a dictionary containing the similarity scores.

    Definition: a document's 'plan' is the headings and subheadings of the document in
                order.

    Example Usage:
    >>> url_1 = 'https://en.wikipedia.org/wiki/Simulated_annealing'
    >>> url_2 = 'https://en.wikipedia.org/wiki/Dual-phase_evolution'
    >>> doc_1 = await extract_plan_and_content_wikipedia(url_1)
    >>> doc_2 = await extract_plan_and_content_wikipedia(url_2)
    >>> compare_plan = compare_documents_sections(doc_1, doc_2, None)
    """
    return _compare_documents(document1, document2, compare_on="section")


def compare_documents_content(
    document1: str | Path | Dict[str, Any],
    document2: str | Path | Dict[str, Any],
) -> Dict[str, Any]:
    """This function takes two documents, a comparison method, compares the sections
    of the documents using the specified method, and returns a dictionary containing
    the section-wise similarity scores.

    Definition: a document's 'content' is the text under the headings and subheadings

    Example Usage:
    >>> url_1 = 'https://en.wikipedia.org/wiki/Simulated_annealing'
    >>> url_2 = 'https://en.wikipedia.org/wiki/Dual-phase_evolution'
    >>> doc_1 = await extract_plan_and_content_wikipedia(url_1)
    >>> doc_2 = await extract_plan_and_content_wikipedia(url_2)
    >>> compare_sections = compare_documents_content(doc_1, doc_2, None)
    """
    # TODO - do we really need method? Or can we just do every metric every time?
    return _compare_documents(document1, document2, compare_on="content")


async def extract_plan_and_content(input: str | Path, doc_type: str) -> Dict[str, Any]:
    """Extract plans and content for a range of doc_types. Write ouputs to individual files.
    Return a dictionary containing the plan and content for the input."""
    logger.info(f"\n\tExtracting plan and content for: {input}")
    start = time()
    # Load depending on doc_type
    if doc_type == "wikipedia":
        article_dict = load_wikipedia_url(input)
    elif doc_type == "arxiv":
        article_dict = load_arxiv_paper(input)
    elif doc_type == "patent":
        article_dict = load_patent_file(input)
    else:
        raise ValueError(
            f"doc_type must be one of 'patent', 'wikipedia', or 'arxiv'. "
            f"Received {doc_type}"
        )
    # Divide and create embeddings
    article_dict = await divide_sections_if_too_large(article_dict, doc_type=doc_type)
    num_sections = len(article_dict.keys())
    min_required_sections = 3
    if num_sections < min_required_sections:
        error_msg = (
            f"\n\tInput: {input}"
            f"\n\tInput document is too small. Found {num_sections} sections. We require at "
            f"least {min_required_sections} sections."
            f"\n\tSections Found: {list(article_dict.keys())}"
            f"\n\tSkipping this document and moving onto the next."
        )
        logger.error(error_msg)
        return {"title": str(input), "content": error_msg}
    plan_json = generate_embeddings_plan_and_section_content(
        article_dict, doc_type=doc_type
    )
    # Write to file
    output_dir = Path(f"output/{doc_type}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / Path(f"{plan_json['title']}.json")
    with open(output_file, "w") as file:
        json.dump(plan_json, file, indent=4)
    # Calculate time taken
    minutes, seconds = divmod(round(time() - start, 3), 60)
    elapsed = (
        f"{str(int(minutes)) + 'mins' if minutes else ''}"
        f" {str(round(seconds, 1)) + 's'}"
    )
    logger.info(
        f"\n\tSuccessfully extracted plan and content for {input}"
        f"\n\tWritten to file: {output_file.absolute()}"
        f"\n\tTime taken: {elapsed}"
    )
    return plan_json


async def extract_plans_and_content(
    inputs: List[str | Path], doc_type: str
) -> List[Dict[str, Any]]:
    """Extract plans and content for a list of inputs. Write ouputs to individual files.
    Return a list of dictionaries containing the plan and content for each input."""
    return [
        await extract_plan_and_content(input, doc_type=doc_type) for input in inputs
    ]


async def extract_plan_and_content_arxiv(path: str | Path) -> Dict[str, Any]:
    """Given a path to an arxiv pdf on disk, return a dictionary with the title, abstract,
    plan and associated embeddings.

    Note: due to the async nature of the function, it must be run using either asyncio.run()
    if called from a script, or using `await` if called from another async function or jupyter
    notebook.

    Example Usage:
    >>> import asyncio
    >>> path = "path/to/arxiv.pdf"
    >>> plan_json = asyncio.run(extract_plan_and_content_arxiv(path))
    """
    return await extract_plan_and_content(path, doc_type="arxiv")


async def extract_plan_and_content_wikipedia(url: str) -> Dict[str, Any]:
    """Given a Wikipedia URL, returns a dictionary with the title, abstract, plan and
    associated embeddings.

    Note: due to the async nature of the function, it must be run using either asyncio.run()
    if called from a script, or using `await` if called from another async function or jupyter
    notebook.

    Example Usage:
    >>> import asyncio
    >>> url = "https://en.wikipedia.org/wiki/Dual-phase_evolution"
    >>> plan_json = asyncio.run(extract_plan_and_content_wikipedia(url))
    """
    return await extract_plan_and_content(url, doc_type="wikipedia")


async def extract_plan_and_content_patent(patent_file: str | Path) -> Dict[str, Any]:
    """Given a path to a patent file on disk, return a dictionary with the title, abstract,
    plan and associated embeddings.

    Note: due to the async nature of the function, it must be run using either asyncio.run()
    if called from a script, or using `await` if called from another async function or jupyter
    notebook.

    Example Usage:
    >>> import asyncio
    >>> path = "path/to/patent.txt"
    >>> plan_json = asyncio.run(extract_plan_and_content_patent(path))
    """
    return await extract_plan_and_content(patent_file, doc_type="patent")


if __name__ == "__main__":
    wikipedia_articles = [
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
        "https://en.wikipedia.org/wiki/Dual-phase_evolution",
        "https://en.wikipedia.org/wiki/Simulated_annealing",
        "https://en.wikipedia.org/wiki/Tessellation",
        "https://en.wikipedia.org/wiki/Climate_change",
        "https://en.wikipedia.org/wiki/DNA_nanotechnology",
        "https://en.wikipedia.org/wiki/Self-driving_car",
        "https://en.wikipedia.org/wiki/Unmanned_aerial_vehicle",
        "https://en.wikipedia.org/wiki/2022%E2%80%932023_food_crises",
        "https://en.wikipedia.org/wiki/Economic_impacts_of_climate_change",
    ]
    for wiki in wikipedia_articles:
        asyncio.run(extract_plan_and_content_wikipedia(wiki))

    patents = list(Path("data/patents").glob("*"))
    # patents = [
    #     "data/patents/MICROWAVE TURNTABLE CONVECTION HEATER.txt",
    #     "data/patents/PHARMACEUTICAL COMPOSITIONS OF GALLIUM COMPLEXES OF 3-HYDROXY-4-PYRONES.txt",
    # ]
    for patent in patents:
        asyncio.run(extract_plan_and_content_patent(patent))

    arxiv = list(Path("data/arxiv").glob("*"))
    for arx in arxiv:
        asyncio.run(extract_plan_and_content_arxiv(arx))
