from setuptools import setup, find_packages

setup(
    name="llm-excitement",
    version="0.1.0",
    description="Identifying interpretable features in LLMs using SAEs and Gemma Scope 2",
    author="AlonKapln",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "huggingface-hub>=0.19.0",
        "safetensors>=0.4.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "einops>=0.7.0",
    ],
    python_requires=">=3.8",
)
