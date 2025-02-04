from setuptools import find_packages, setup

setup(
    name="langchain-playground",
    version="0.1.0",
    author="Teron",
    author_email="teron131@gmail.com",
    description="LangChain Playground for my personal use.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/teron131/langchain-playground",
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "*test*",
            "*.__pycache__",
            "__pycache__",
            "*.cache",
            ".cache",
            "cache",
            "images",
            "audio",
            "pdfs",
            "scratch",
            "build",
            "langchain_playground.egg-info",
            "langchain_playground/AG2/coding",
        ]
    ),
    package_data={
        "": [
            "!*.log",
            "!*.png",
            "!*.jpg",
            "!*.pdf",
            "!*.json",
            "!*.pem",
            "!.env",
            "!.cache",
        ],
    },
    install_requires=[
        # Core Dependencies
        "numpy",
        "pandas",
        "pydantic",
        "python-dotenv",

        # LangChain Framework
        "langchain",
        "langchain-community",
        "langchain-google-genai",
        "langchain-openai",
        "langgraph",
        "langsmith",
        "openai",

        # Providers
        "fal-client",
        "google-genai",
        "replicate",

        # Media Processing
        "openai-whisper",
        "optimum",
        "pillow",
        "pytubefix",
        "tiktoken",
        "torch",
        "transformers",

        # Document Processing
        "docling",
        "opencc-python-reimplemented",
        "tavily-python",

        # Utilities
        "httpx",
        "ipython",
        "more-itertools",
        "tqdm",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
