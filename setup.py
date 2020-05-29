from setuptools import setup, find_packages

setup(
    name="allennlp_beaker",
    version="0.0.1",
    description=(
        "An interactive AllenNLP plugin for submitting training jobs to beaker"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="allennlp NLP beaker",
    url="https://github.com/allenai/allennlp-beaker",
    author="Allen Institute for Artificial Intelligence",
    author_email="allennlp@allenai.org",
    license="Apache",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"],),
    install_requires=["allennlp", "click", "PyYAML", "click-spinner"],
    entry_points={"console_scripts": ["allennlp-beaker=allennlp_beaker.__main__:run"]},
    python_requires=">=3.6.1",
)
