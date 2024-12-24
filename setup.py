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
            "*.ipynb",
            "*.__pycache__",
            "*.cache",
            "images",
            "audio",
            "pdfs",
            "scratch",
        ]
    ),
    package_data={
        "": [
            "!*.log",
            "!*.png",
            "!*.jpg",
            "!*.pdf",
            "!*.json",
            "!.env",
            "!.cache",
        ],
    },
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-openai",
        "langchain-together",
        "gradio",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)