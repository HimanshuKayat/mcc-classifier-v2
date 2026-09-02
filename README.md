MCC Classifier v2

A semantic Merchant Category Code (MCC) classification pipeline for identifying the commercial nature of entities and assigning the most relevant MCCs from a fixed MCC catalogue.

The pipeline combines Llama 3.1 8B via Ollama for entity understanding and final MCC ranking with Sentence Transformers for semantic retrieval and similarity scoring.

Overview

For each input entity/article name, the classifier performs four major stages:

Input Excel
    │
    ▼
┌──────────────────────────────┐
│ 1. Entity Understanding      │
│ Llama 3.1 8B + Entity Prompt │
└──────────────┬───────────────┘
               │
               ▼
       Entity + Commercial
            Profiles
               │
               ▼
┌──────────────────────────────┐
│ 2. Semantic Retrieval        │
│ all-MiniLM-L6-v2             │
│ Retrieve Top-20 MCCs         │
└──────────────┬───────────────┘
               │
               ▼
          Top-20 MCCs
               │
               ▼
┌──────────────────────────────┐
│ 3. LLM Ranking               │
│ Llama 3.1 8B                 │
│ Select and rank Top-5        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 4. Confidence Scoring        │
│ 80% semantic similarity      │
│ 20% retrieval similarity     │
└──────────────┬───────────────┘
               │
               ▼
       Output Excel

The important design principle is that entity understanding and MCC selection are separated. The entity/commercial profile is generated independently of the MCC candidates, then semantic retrieval narrows the search space, and the LLM makes the final Top-5 ranking only from those retrieved candidates.

Key Features

Two-stage LLM workflow

Stage 1: understand the entity and create a structured commercial profile.

Stage 2: rank the best MCCs from retrieved candidates.

Semantic retrieval using sentence-transformers/all-MiniLM-L6-v2.

Top-20 candidate retrieval before LLM classification.

Top-5 MCC predictions with a concise reason for each prediction.

Confidence score calculated from semantic and retrieval similarity.

Strict JSON parsing and validation of LLM responses.

Candidate enforcement: the LLM cannot return an MCC outside the retrieved Top-20.

Duplicate/rank validation for the final five predictions.

Excel input/output using pandas and openpyxl.

Resume support: rows already marked Completed are skipped when the output file exists.

Periodic checkpointing every 50 processed rows.

Per-row error capture in the output workbook.

Repository Structure

mcc-classifier-v2-main/
│
├── data/
│   ├── articles_metadata.xlsx       # Main input dataset
│   ├── mcc_codes.json               # MCC catalogue / profiles
│   ├── test_cases.json              # Small test data / legacy test fixture
│   └── sample_inputs.csv            # Placeholder sample file
│
├── models/
│   ├── __init__.py
│   └── llama_model.py               # Ollama HTTP client
│
├── prompts/
│   ├── entity_prompt.py             # Entity/commercial profile prompt
│   ├── mcc_prompt.py                # Top-5 MCC ranking prompt
│   ├── prompt_builder.py            # Legacy prompt builder
│   ├── classification_prompt.txt    # Legacy/unused prompt file
│   └── system_prompt.txt            # Legacy/unused prompt file
│
├── retriever/
│   ├── __init__.py
│   └── embedding_retriever.py       # MCC retrieval + similarity calculation
│
├── scripts/
│   ├── classifier.py                # Main classification orchestration
│   ├── confidence_scorer.py         # Confidence calculation
│   └── test_llama.py                # Legacy test script
│
├── tests/
│   └── test_classifier.py           # Placeholder test file
│
├── build_embeddings.py              # Pre-compute MCC embeddings
├── main.py                          # Batch Excel processing entry point
├── parser.py                        # LLM JSON parsing and validation
├── config.py                        # Currently empty / reserved
├── requirements.txt                 # Python dependencies
└── .gitignore

Processing Pipeline

1. Entity Understanding

MCCClassifier.classify() first sends the article/entity name to Llama 3.1 8B using EntityPromptBuilder.

The model is explicitly instructed not to classify the entity into an MCC at this stage. It instead returns two structures:

entity_profile

commercial_profile

The entity profile contains fields such as:

Entity name

Entity type

Summary

Primary business

Industry

Products/services

Target customers

Business model

Parent company

Country

Keywords

Aliases

The commercial profile is designed specifically for downstream semantic comparison and contains:

Commercial activity

Primary offering

Delivery method

Customer type

Revenue model

Business context

This separation reduces the risk of the model simply matching an entity name to an MCC based on surface-level wording.

2. Semantic MCC Retrieval

EmbeddingRetriever converts the structured entity profile into text and embeds it with:

sentence-transformers/all-MiniLM-L6-v2

It then calculates cosine similarity against the pre-computed embeddings for the MCC catalogue and retrieves the Top 20 candidates.

The catalogue currently contains 305 MCC profiles in data/mcc_codes.json.

Each MCC profile contains fields including:

{
  "mcc": "0742",
  "industry": "Veterinary Services",
  "category": "Agriculture",
  "description": "...",
  "keywords": [],
  "aliases": []
}

3. LLM Top-5 Ranking

The Top-20 retrieved candidates are passed to MCCPromptBuilder along with the entity profile.

Llama is instructed to:

Select exactly five MCCs.

Select only from the supplied 20 candidates.

Never invent or modify an MCC.

Rank the five from best to least appropriate.

Focus on the entity's actual primary commercial activity.

Provide one concise reason for every selected MCC.

The parser subsequently validates that all five returned MCCs were actually present in the Top-20 retrieval set.

4. Confidence Scoring

For every selected MCC, the pipeline calculates an independent semantic similarity between the commercial profile and the selected MCC profile.

Final confidence is calculated as:

Confidence = (0.80 × Semantic Similarity)
           + (0.20 × Retrieval Similarity)

Both inputs are clamped to the [0, 1] range.

The 80/20 weighting is implemented in scripts/confidence_scorer.py.

Input Data

The default input is:

data/articles_metadata.xlsx

The current workbook contains an article column, which is the only input field directly used for classification by main.py.

Other input columns are preserved in the output workbook.

Expected minimum column:

article

Example:

article

page_id

categories

Starbucks

...

...

...

...

...

Output

The default output path is:

outputs/articles_metadata_output.xlsx

The original input columns are retained and the classifier appends entity, prediction, and processing fields.

Entity columns

Entity Name
Entity Type
Entity Summary
Primary Business
Industry
Products/Services
Target Customers
Business Model
Parent Company
Country
Keywords
Aliases

MCC columns

For each of the five predictions:

MCC Top N
MCC Top N Industry
MCC Top N Semantic Similarity
MCC Top N Retrieval Similarity
MCC Top N Confidence
MCC Top N Reason

where N is 1 through 5.

Processing columns

Processing Status
Processing Error

A successfully processed row receives:

Processing Status = Completed

Failed rows receive:

Processing Status = Failed

and the exception message is written to Processing Error.

Requirements

Python dependencies are listed in requirements.txt:

ollama
requests
pandas
openpyxl
tqdm
sentence-transformers
scikit-learn

The current implementation communicates with Ollama through its HTTP API using requests. The ollama Python package is listed as a dependency but is not directly used by the current classifier implementation.

Installation

1. Clone the repository

git clone <repository-url>
cd mcc-classifier-v2-main

2. Create a virtual environment

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

Ollama Setup

The classifier expects an Ollama server running locally at:

http://localhost:11434/api/generate

The default model is:

llama3.1:8b

Install Ollama separately, start the Ollama service, and make sure the model is available locally.

Verify the model before running the pipeline:

ollama list

If the model is not installed, pull it with:

ollama pull llama3.1:8b

You can also test the Ollama endpoint independently before running the full classifier.

Build MCC Embeddings

Before running main.py, generate the MCC embedding file:

python build_embeddings.py

This script:

Loads sentence-transformers/all-MiniLM-L6-v2.

Loads data/mcc_codes.json.

Converts every MCC profile to text.

Generates an embedding for every MCC.

Saves the profiles and embeddings to:

data/mcc_embeddings.pkl

The embedding file is required by EmbeddingRetriever.

You generally only need to rebuild it when data/mcc_codes.json changes or when you intentionally change the embedding model.

Run the Classifier

From the repository root:

python main.py

The program will:

Load data/articles_metadata.xlsx.

Create the output directory if required.

Resume from outputs/articles_metadata_output.xlsx if it already exists.

Skip rows whose status is Completed.

Process each article sequentially.

Save progress every 50 processed rows.

Perform a final save after all rows are processed.

Resume Behaviour

The batch processor is designed to survive interruptions.

If this file exists:

outputs/articles_metadata_output.xlsx

main.py loads it instead of starting from the original input workbook.

Rows with:

Processing Status = Completed

are skipped.

Rows that previously failed are eligible to be attempted again on a subsequent run.

Progress is checkpointed every 50 processed rows through the SAVE_EVERY setting in main.py.

Configuration

The current runtime settings are defined directly near the top of main.py:

INPUT_FILE = "data/articles_metadata.xlsx"
OUTPUT_FILE = "outputs/articles_metadata_output.xlsx"
ARTICLE_COLUMN = "article"
SAVE_EVERY = 50
DEBUG = False

Change the input file

INPUT_FILE = "data/my_input.xlsx"

Change the output file

OUTPUT_FILE = "outputs/my_output.xlsx"

Change the input article column

ARTICLE_COLUMN = "article"

Enable retrieval debugging

DEBUG = True

When enabled, the classifier prints the Top-20 retrieved MCCs and their retrieval similarity scores.

config.py currently exists but is empty. Runtime configuration is currently controlled in main.py and constructor defaults in the individual modules.

LLM Configuration

The Ollama client is implemented in:

models/llama_model.py

Its defaults are:

Model: llama3.1:8b
Endpoint: http://localhost:11434/api/generate

The classifier currently instantiates LlamaModel() without passing custom values, so changing the default constructor values or extending MCCClassifier would be required to make model/host configuration externally configurable.

JSON Parsing and Validation

parser.py protects the pipeline from malformed LLM responses.

Entity result validation

The entity response must contain:

entity_profile
commercial_profile

Prediction validation

The MCC response must contain exactly five predictions.

Validation checks include:

Predictions are a list.

Exactly five predictions are returned.

Ranks are exactly 1–5.

Every MCC is non-empty.

Every MCC belongs to the retrieved Top-20.

No MCC is duplicated.

Every prediction contains a reason.

If validation fails, the row is marked Failed and the exception is recorded in the output workbook.

Confidence Model

The confidence scorer intentionally does not ask the LLM to provide a confidence value.

Instead, confidence is computed deterministically from embedding similarities:

SEMANTIC_WEIGHT = 0.80
RETRIEVAL_WEIGHT = 0.20

This produces a score that reflects both:

how strongly the independent commercial profile matches the selected MCC, and

how strongly the entity profile initially retrieved that MCC.

This design avoids relying on self-reported LLM confidence.

Important Files

File

Purpose

main.py

Batch processing, Excel I/O, resume logic, checkpointing

scripts/classifier.py

End-to-end classification orchestration

models/llama_model.py

Ollama/Llama HTTP interface

prompts/entity_prompt.py

Entity and commercial profile extraction prompt

prompts/mcc_prompt.py

Top-20 → Top-5 MCC ranking prompt

retriever/embedding_retriever.py

Embedding retrieval and semantic similarity

scripts/confidence_scorer.py

Deterministic confidence calculation

parser.py

JSON extraction and result validation

build_embeddings.py

MCC embedding generation

data/mcc_codes.json

MCC reference catalogue

data/articles_metadata.xlsx

Default classification input

Legacy / Unused Components

The repository contains several files that do not appear to participate in the current main.py → MCCClassifier execution path:

prompts/prompt_builder.py

Contains an older single-stage entity/classification prompt design. The current pipeline uses EntityPromptBuilder and MCCPromptBuilder instead.

prompts/classification_prompt.txt

Currently empty and not referenced by the active pipeline.

prompts/system_prompt.txt

Contains an older single-MCC classification prompt and is not referenced by the active pipeline.

scripts/test_llama.py

The script uses an older MCCClassifier.classify() calling convention (merchant_name and description) that no longer matches the current classifier signature. It should be updated before being used as a test entry point.

tests/test_classifier.py

Currently empty. There is no implemented automated test suite in the repository as provided.

config.py

Currently empty. Configuration has not yet been centralized into this module.

These files can be cleaned up or modernized if the repository is being prepared for production use.

Current Execution Path

The active production path is effectively:

main.py
   │
   ▼
scripts/classifier.py
   │
   ├── models/llama_model.py
   ├── prompts/entity_prompt.py
   ├── retriever/embedding_retriever.py
   ├── prompts/mcc_prompt.py
   ├── parser.py
   └── scripts/confidence_scorer.py

The embedding build path is:

build_embeddings.py
   │
   ├── data/mcc_codes.json
   └── sentence-transformers/all-MiniLM-L6-v2
            │
            ▼
   data/mcc_embeddings.pkl

Example End-to-End Run

A clean setup would typically be:

# 1. Create environment
python -m venv .venv

# 2. Activate environment
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure Ollama has the required model
ollama pull llama3.1:8b

# 5. Generate MCC embeddings
python build_embeddings.py

# 6. Run classification
python main.py

The resulting workbook will be written to:

outputs/articles_metadata_output.xlsx

Performance Considerations

The pipeline performs multiple model operations per input row:

Llama generation for entity understanding.

Sentence-transformer encoding for entity retrieval.

Llama generation for MCC ranking.

Sentence-transformer encoding for each selected MCC comparison.

Consequently, processing a large workbook can take substantial time, particularly when Ollama is running an 8B model on CPU or limited GPU hardware.

The implementation processes rows sequentially and periodically saves progress to reduce the impact of interruptions.

Data and Artifact Notes

data/mcc_embeddings.pkl is a generated artifact rather than a source dataset. If it is absent, build_embeddings.py must be run before classification.

Generated outputs under outputs/ are ignored by Git through .gitignore.

The repository's input workbook and MCC JSON catalogue are currently committed under data/.

Troubleshooting

FileNotFoundError: data/mcc_embeddings.pkl

Run:

python build_embeddings.py

Ollama connection error

Make sure Ollama is running and the endpoint is reachable at:

http://localhost:11434/api/generate

Then verify:

ollama list

Model not found

Pull the configured model:

ollama pull llama3.1:8b

JSON parsing failure

The LLM response must contain valid JSON. The parser attempts to extract a JSON object, but malformed or incomplete model output will cause the current row to be marked Failed.

Prediction validation failure

The LLM must return exactly five unique MCCs selected from the retrieved Top-20. Returning an MCC outside that set, duplicating a code, using incorrect ranks, or omitting a reason causes validation to fail.

Existing output appears to resume unexpectedly

Delete or rename:

outputs/articles_metadata_output.xlsx

if you intentionally want to start the batch from the original input workbook again.

Development Recommendations

For the next iteration of the project, the most useful improvements would be:

Move runtime configuration from main.py into config.py or environment variables.

Add pinned dependency versions for reproducibility.

Add automated unit tests for parsing, retrieval, confidence scoring, and output writing.

Update scripts/test_llama.py to match the current classify(page_name) API.

Add a small deterministic test fixture that does not require a running LLM.

Add structured logging rather than relying only on console output.

Consider caching entity-level results to avoid repeating LLM work after partial failures.

Add an explicit version/metadata record for the embedding model and MCC catalogue used to create mcc_embeddings.pkl.

Consider moving model names, Ollama host, input path, output path, and save frequency into a single configuration layer.

Design Summary

The classifier follows this conceptual model:

Entity Name
    │
    ▼
LLM Entity Understanding
    │
    ├───────────────┐
    ▼               ▼
Entity Profile   Commercial Profile
    │               │
    ▼               │
Semantic Top-20     │
Retrieval           │
    │               │
    └───────┬───────┘
            ▼
      LLM Top-5 Ranking
            │
            ▼
 Independent Semantic
 Similarity Calculation
            │
            ▼
   80/20 Confidence Score
            │
            ▼
        Final Output

The key separation is:

Understand → Retrieve → Rank → Score

rather than asking the LLM to directly choose an MCC from the complete catalogue.
