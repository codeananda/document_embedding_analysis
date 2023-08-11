# 📄 Document Embedding Analysis
*Harnessing embeddings for direct content comparison analysis.*

## 💻 How to Run the Code

#### 1. Download code + create env
```
$ git clone https://github.com/codeananda/document_extraction.git
$ cd document_extraction
$ pip install -r requirements.txt
```

#### 2. Set your OpenAI key

1. Go to https://platform.openai.com/account/api-keys to create a new key (if you don't have one already).
2. Rename `.env_template` to `.env`
3. Add your key in next to `OPENAI_API_KEY`

## 🤔 Questions

### What do I need to modify if I want to change `max_section_length` to > 512

You must remove/comment out parts of the code that only accept a max of 512 tokens as input: e5-base-v2 and MAUVE.

* Remove e5-base-v2 embeddings code

This is found in `_gen_embed_section_content` and `generate_embeddings_plan_and_section_content` and is defined like so

```python
embed_e5 = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2", encode_kwargs={"normalize_embeddings": True}
)
```

Comment that out and all references to it. 

* Remove all references to `_embedding_2`
* Remove mauve calculations (Crtl + F and search 'mauve' to find them)

### How do I manually edit text extracted from pdfminer?

* Run `load_arxiv_paper(path_to_paper)`
* Write output to disk using `json.dump`
* Modify the `Content` key.
* Load content from disk using `json.load`
* See `extract_plan_and_content` for functions to pass to next 
