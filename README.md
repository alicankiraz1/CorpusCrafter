# CorpusCrafter

<img src="https://github.com/alicankiraz1/CorpusCrafter/blob/main/CorpusCrafter.png" width="500" height="500">

Smith your data for tomorrow's intelligence!

![GitHub License](https://img.shields.io/badge/license-MIT-green)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2.0%2B-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-1.25%2B-red)


## 📚 Overview

CorpusCrafter is an advanced tool designed to transform PDF documents into structured AI training datasets following OpenAI's system/user/assistant format. Created specifically for researchers, data scientists, and AI developers, this tool streamlines the process of preparing high-quality conversational data for fine-tuning language models.

CorpusCrafter intelligently extracts content from books, academic papers, and technical documents, divides them into meaningful chunks, and automatically generates relevant questions for each text segment. The result is a perfectly formatted dataset ready for AI model training and evaluation.


## ✨ Key Features

### 🔍 Intelligent PDF Extraction
- **Selective Content Extraction**: Automatically filters out irrelevant elements such as cover pages, prefaces, indexes, references, and footnotes
- **Empty Page Detection**: Skips blank pages and pages containing only images to prevent unnecessary data generation
- **Text Cleaning**: Cleans consecutive spaces, excess line breaks, and other formatting to produce consistent text

### 🧠 Advanced Text Processing
- **Language Detection**: Automatically detects the document language and applies language-specific processing
- **Heading and Question Analysis**: Intelligently identifies headings, filters questions, and preserves the structural integrity of the text
- **Unicode Normalization**: Applies Unicode NFKC normalization for proper text processing
- **Linguistic Analysis**: Optionally uses the spaCy library for advanced linguistic analysis
- **Named Entity Recognition**: Identifies and extracts named entities (people, organizations, locations) from text
- **Sentiment Analysis**: Determines the emotional tone and sentiment polarity of text segments
- **Text Summarization**: Creates concise summaries of longer text passages
- **Keyword Extraction**: Automatically identifies and extracts key terms and concepts

### 📊 Semantic Chunking
- **Context Preservation**: Divides text into meaningful chunks while preserving sentence and paragraph integrity
- **Customizable Dimensions**: Chunk size and overlap amount can be adjusted by the user
- **Various Algorithm Options**: Recursive character splitting, token-based splitting, and sentence transformers-based splitting options
- **Semantic Similarity Clustering**: Groups related content based on meaning rather than just proximity
- **Multiple Clustering Algorithms**: Supports K-means, Agglomerative, DBSCAN, and Spectral clustering methods
- **Hierarchical Chunking**: Preserves document structure based on headings and sections
- **Sliding Window Approach**: Creates overlapping chunks with configurable window size and step

### 💡 AI-Ready Dataset Generation
- **OpenAI Format Compliance**: Creates datasets in the system/user/assistant format required for fine-tuning OpenAI models
- **Context-Sensitive Questions**: Generates questions that capture the main idea and important concepts of each text chunk
- **Multilingual Support**: Language-specific question generation prompts for English, Turkish, and other languages
- **Model Flexibility**: Ability to use different OpenAI models (GPT-3.5, GPT-4o, GPT-4o-mini) for question generation
- **Advanced Question Types**: Generates diverse question formats including open-ended, multiple-choice, true/false, and fill-in-the-blank
- **Customizable Difficulty Levels**: Adjustable question difficulty from easy to expert
- **Bloom's Taxonomy Integration**: Questions target different cognitive levels (remembering, understanding, applying, analyzing, evaluating, creating)
- **Domain-Specific Questions**: Specialized question generation for medical, legal, technical, and academic domains

### 🛠️ Advanced Tool Features
- **Error Management**: Intelligent retry strategy and exponential backoff for API errors
- **Progress Tracking**: Real-time progress indicator with tqdm
- **Comprehensive Logging**: Configurable logging system for detailed debugging and tracking
- **Optional Dependencies**: Minimum requirements for core functionality, optional packages for advanced features


## 🚀 Installation

### Requirements
- Python 3.10 or higher
- LangChain 0.2.0 or higher
- PyPDF2 3.0 or higher
- pandas 2.0 or higher
- openai 1.25 or higher

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/username/pdf-to-csv-dataset.git
cd pdf-to-csv-dataset

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
# Linux/MacOS
export OPENAI_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:OPENAI_API_KEY="your_api_key_here"

# Windows (CMD)
set OPENAI_API_KEY=your_api_key_here

```

### Optional Dependencies

You can also install these packages for advanced features:

```bash
# For language detection
pip install langdetect

# For advanced linguistic analysis
pip install spacy
python -m spacy download en_core_web_sm   # English language model
python -m spacy download tr_core_news_sm  # Turkish language model

# For advanced NLP features
pip install transformers sentence-transformers torch nltk yake

# For smart text chunking
pip install scikit-learn networkx gensim
```


## 📖 Usage

### Basic Usage

CorpusCrafter offers straightforward commands for quick PDF-to-dataset conversion:

```bash
# Process the first PDF in the input directory
python -m corpuscrafter

# Process a specific PDF file
python -m corpuscrafter --file your_document.pdf

# Use a specific OpenAI model
python -m corpuscrafter --model gpt-4o

# Specify input and output directories
python -m corpuscrafter --input your_pdfs --output your_datasets

# Auto-detect document language
python -m corpuscrafter --language auto

```

### Advanced Usage

For more sophisticated use cases, CorpusCrafter offers extensive customization options. These options allow you to control every aspect of the conversion process:

```bash
# Full configuration example with all major parameters
python -m corpuscrafter \
  --input custom_input_dir \
  --output custom_output_dir \
  --file academic_paper.pdf \
  --model gpt-4o \
  --splitter recursive \
  --chunk-size 1500 \
  --chunk-overlap 150 \
  --language en \
  --system-prompt "You are an expert tutor helping students understand complex topics." \
  --temperature 0.7 \
  --max-tokens 200 \
  --verbose

# Process all PDFs in a directory using batch mode
python -m corpuscrafter --batch --input pdf_collection --output datasets

# Use custom separators for more precise text splitting
python -m corpuscrafter --file technical_manual.pdf --custom-separators "\n\n" "\n" ". "

# Preserve specific elements that would normally be filtered
python -m corpuscrafter --file textbook.pdf --preserve-headings --preserve-questions

# Apply OCR for scanned documents (requires optional dependencies)
python -m corpuscrafter --file scanned_document.pdf --enable-ocr

# Export in multiple formats simultaneously
python -m corpuscrafter --file document.pdf --formats csv jsonl

# Apply custom filtering to target specific content
python -m corpuscrafter --file report.pdf --filter-regex "Chapter \d+|Section \d+\.\d+"

# Limit the number of chunks or specify page ranges
python -m corpuscrafter --file large_book.pdf --max-chunks 100 --pages 10-50

# Use a local embedding model instead of API calls (reduces costs)
python -m corpuscrafter --file document.pdf --use-local-embeddings
```

### Advanced Feature Usage

CorpusCrafter's new advanced features provide even more powerful options for customization:

```bash
# Advanced Question Generation
python -m corpuscrafter \
  --file textbook.pdf \
  --use-advanced-question-generator \
  --question-types open_ended multiple_choice true_false \
  --question-difficulty medium \
  --cognitive-level understanding \
  --questions-per-chunk 3 \
  --include-answers \
  --domain medical

# Advanced NLP Processing
python -m corpuscrafter \
  --file research_paper.pdf \
  --use-advanced-nlp \
  --enable-ner \
  --enable-sentiment \
  --enable-summarization \
  --enable-keyword-extraction \
  --nlp-model-type transformer \
  --use-gpu

# Smart Text Chunking
python -m corpuscrafter \
  --file technical_manual.pdf \
  --use-smart-chunking \
  --semantic-chunking \
  --clustering-method kmeans \
  --n-clusters 5 \
  --similarity-threshold 0.7 \
  --chunk-method semantic_similarity

# Combined Advanced Features
python -m corpuscrafter \
  --file academic_paper.pdf \
  --use-advanced-question-generator \
  --question-types open_ended analytical \
  --use-advanced-nlp \
  --enable-summarization \
  --use-smart-chunking \
  --semantic-chunking \
  --clustering-method agglomerative
```

When working with advanced options, you can combine multiple parameters to create a highly customized conversion pipeline. CorpusCrafter's modular architecture ensures that each component (extraction, processing, chunking, question generation) can be independently configured to match your specific requirements.
The verbose mode (--verbose) provides detailed logs of each processing step, which is particularly useful when fine-tuning your parameters or debugging issues with complex documents.



## 📊 Example Output

The tool generates a CSV file with the following structure:

| chunk_id | system | user              | assistant                     |
|----------|--------|-------------------|-------------------------------|
| uuid-1   |        | Text chunk 1…     | Question about chunk 1?       |
| uuid-2   |        | Text chunk 2…     | Question about chunk 2?       |
| …        |        | …                 | …                             |

### Advanced Output Examples

With advanced features enabled, the output can include more sophisticated content:

#### Advanced Question Generation

```csv
chunk_id,system,user,assistant
uuid-1,,The mitochondrion is often referred to as the powerhouse of the cell...,"{""question"":""What is the primary function of mitochondria in cells?"",""type"":""open_ended"",""difficulty"":""medium"",""cognitive_level"":""understanding""}"
uuid-2,,Photosynthesis is the process by which plants convert light energy into chemical energy...,"{""question"":""Which of the following is NOT a product of photosynthesis?"",""type"":""multiple_choice"",""options"":{""A"":""Oxygen"",""B"":""Glucose"",""C"":""Carbon dioxide"",""D"":""ATP""},""answer"":""C"",""difficulty"":""hard"",""cognitive_level"":""analyzing""}"
```

#### Advanced NLP Processing

```csv
chunk_id,system,user,assistant,entities,sentiment,keywords,summary
uuid-1,,The Supreme Court ruled in favor of the plaintiff...,Question about the ruling?,"{""Supreme Court"":""ORG"",""plaintiff"":""PERSON""}","{""polarity"":0.2,""subjectivity"":0.4}","{""Supreme Court"":0.8,""ruling"":0.7,""plaintiff"":0.6}","The Supreme Court made a favorable ruling for the plaintiff."
```

#### Smart Text Chunking

```csv
chunk_id,system,user,assistant,cluster_id,similarity_score,hierarchy_level
uuid-1,,Introduction to machine learning...,Question about ML intro?,1,0.85,1
uuid-2,,Types of supervised learning algorithms...,Question about supervised learning?,1,0.82,2
uuid-3,,Quantum computing fundamentals...,Question about quantum computing?,2,0.91,1
```

---

## 🔧 Available Options

| Option              | Short | Description                                                     | Default      |
|---------------------|-------|-----------------------------------------------------------------|--------------|
| `--input`           | `-i`  | Input directory that contains PDF files                         | `input`      |
| `--output`          | `-o`  | Output directory for generated CSV files                        | `output`     |
| `--file`            | `-f`  | Specific PDF file to process (optional)                         | —            |
| `--model`           | `-m`  | OpenAI model to use                                             | `gpt-4o-mini`|
| `--splitter`        | `-s`  | Text-splitting algorithm                                        | `recursive`  |
| `--chunk-size`      | `-cs` | Chunk size in characters                                        | `2000`       |
| `--chunk-overlap`   | `-co` | Overlap between chunks in characters                            | `200`        |
| `--language`        | `-l`  | Document language (`auto` for automatic detection)              | `auto`       |
| `--list-models`     | `-lm` | List available OpenAI models                                    | —            |
| `--list-splitters`  | `-ls` | List available text-splitting algorithms                        | —            |
| `--list-languages`  | `-ll` | List supported languages                                        | —            |
| `--version`         | `-v`  | Show version information                                        | —            |

### Advanced Feature Options

| Option                          | Description                                                     | Default      |
|---------------------------------|-----------------------------------------------------------------|--------------|
| `--use-advanced-question-generator` | Enable advanced question generation features                | `False`      |
| `--question-types`              | Types of questions to generate (comma-separated)                | `open_ended` |
| `--question-difficulty`         | Difficulty level of generated questions                         | `medium`     |
| `--cognitive-level`             | Bloom's taxonomy cognitive level for questions                  | `understanding` |
| `--questions-per-chunk`         | Number of questions to generate per text chunk                  | `1`          |
| `--include-answers`             | Include answers with generated questions                        | `False`      |
| `--domain`                      | Specific domain for question generation                         | —            |
| `--use-advanced-nlp`            | Enable advanced NLP processing features                         | `False`      |
| `--enable-ner`                  | Enable Named Entity Recognition                                 | `False`      |
| `--enable-sentiment`            | Enable sentiment analysis                                       | `False`      |
| `--enable-summarization`        | Enable text summarization                                       | `False`      |
| `--enable-keyword-extraction`   | Enable keyword extraction                                       | `False`      |
| `--nlp-model-type`              | Type of NLP model to use (spacy, transformer)                   | `spacy`      |
| `--use-gpu`                     | Use GPU for NLP processing if available                         | `False`      |
| `--use-smart-chunking`          | Enable smart text chunking features                             | `False`      |
| `--semantic-chunking`           | Use semantic similarity for chunking                            | `False`      |
| `--clustering-method`           | Clustering method for semantic chunking                         | `kmeans`     |
| `--n-clusters`                  | Number of clusters for clustering methods                       | `5`          |
| `--similarity-threshold`        | Similarity threshold for semantic similarity chunking           | `0.7`        |
| `--chunk-method`                | Chunking method (semantic_similarity, sliding_window, etc.)     | `semantic_similarity` |


## 🔍 Technical Details

### PDF Extraction Process
The PDF extraction process follows these steps:
- **Initial Filtering:** Uses heuristics to identify cover pages, indexes, and other irrelevant sections
- **Page Analysis:** Evaluates the content density of each page and discards pages without text
- **Text Extraction:** Uses the PyPDF2 library to extract text content
- **Footnote and Reference Filtering:** Identifies and filters numbered footnotes and references

### Text Processing Algorithm
Text processing consists of several steps to enhance text quality:
- **Unicode Normalization:** NFKC normalization to standardize special characters and accents
- **Heading Detection:** Algorithms that identify language-specific heading patterns
- **Question Filtering:** Identifying questions with question marks and language-specific question words
- **Paragraph Structure Preservation:** Combining content under headings to create meaningful paragraphs
- **Whitespace Normalization:** Organizing consecutive spaces and line breaks
- **Named Entity Recognition:** Identifies and classifies named entities in text using transformer models
- **Sentiment Analysis:** Determines emotional tone and polarity of text segments
- **Text Summarization:** Creates concise summaries preserving key information
- **Keyword Extraction:** Extracts important terms and concepts using statistical and ML methods

### Chunking Strategies
The tool offers multiple chunking strategies:
- **Recursive Character Splitting:** Recursively splits text according to specific separators (e.g., paragraphs, sentences)
- **Token-Based Splitting:** Splits text according to token count, suitable for language models
- **Sentence Transformers Splitting:** Uses a special transformer model to preserve sentence boundaries
- **Semantic Similarity Chunking:** Groups text based on meaning rather than just proximity
- **Clustering-Based Chunking:** Uses various clustering algorithms (K-means, Agglomerative, DBSCAN, Spectral)
- **Hierarchical Chunking:** Preserves document structure based on headings and sections
- **Sliding Window Approach:** Creates overlapping chunks with configurable window size and step

### Question Generation Process
Question generation is performed using the OpenAI API as follows:
- **Text Analysis:** Determines the main idea and important concepts of each text chunk
- **Language-Specific Prompts:** Uses language-specific prompts to generate questions
- **Model Parameters:** Controls question variety and length with temperature, top_p, and max_tokens values
- **Error Handling:** Implements retry logic and backoff strategy for API errors
- **Advanced Question Types:** Generates diverse question formats (open-ended, multiple-choice, true/false, fill-in-the-blank)
- **Difficulty Levels:** Adjusts question complexity from easy to expert
- **Cognitive Levels:** Targets different levels of Bloom's taxonomy
- **Domain Specialization:** Customizes questions for specific domains like medical, legal, or academic content



## 🤝 Contributing
We welcome your contributions! Here are several ways you can contribute:

- **Bug Reports:** Report bugs you find through GitHub Issues
- **Feature Requests:** Suggest new features or improvements
- **Code Contributions:** Contribute to the codebase by submitting Pull Requests
- **Documentation Improvements:** Help improve the documentation



## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

### 🙏 Acknowledgments
This tool leverages the following open source projects:
- LangChain - For text splitting utilities
- PyPDF2 - For PDF processing
- OpenAI - For language models
- pandas - For data manipulation
- spaCy - For linguistic analysis (optional)
- Transformers - For advanced NLP capabilities
- Sentence-Transformers - For semantic text processing
- scikit-learn - For clustering algorithms
- NLTK - For natural language processing utilities



## 📝 Frequently Asked Questions

### Which languages does this tool support?
The tool currently has full support for English and Turkish. Thanks to its language detection feature, other languages are also supported at a basic level, although heading and question detection features may not be optimized for these languages.

### Can it process large PDFs?
Yes, the tool is optimized to process large PDFs. It uses intelligent strategies for memory management and can efficiently process large documents.

### What is the API usage cost for GPT-4 or other models?
API usage cost depends on the amount of text processed and the model selected. If you are concerned about costs, you can use the more economical gpt-3.5-turbo model.

### How can I improve the quality of the extracted text?
To improve text quality, you can optionally install the spaCy library and perform more advanced linguistic analysis. Additionally, you can optimize chunking quality by adjusting the --chunk-size and --chunk-overlap parameters.

### Can it process tables and images in PDFs?
The current version primarily focuses on textual content. Tables and images are automatically skipped or interpreted as plain text. Table detection and structured data extraction features may be added in future versions.

### How do I use the advanced question generation features?
Enable advanced question generation with the --use-advanced-question-generator flag and customize the types, difficulty, and cognitive levels of questions using the corresponding parameters.

### What are the benefits of smart text chunking?
Smart text chunking creates more semantically coherent chunks by grouping related content together, resulting in better context preservation and more meaningful questions.

### Does the tool support GPU acceleration?
Yes, for advanced NLP processing and smart text chunking, you can enable GPU acceleration with the --use-gpu flag if you have a compatible GPU available.

### Can I customize the output format?
Yes, you can export data in multiple formats and customize the structure of the output files. The default format is CSV, but you can also export to JSONL and other formats.

### How do I handle documents in multiple languages?
Use the --language auto option to automatically detect the document language, or specify the language explicitly with --language [code] if you know it in advance.
