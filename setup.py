import pathlib
import setuptools # type: ignore

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setuptools.setup( # type: ignore
    name="allib",
    version="0.0.1",
    description="A type-safe Active Learning library",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Michiel Bron",
    author_email="m.p.bron@uu.nl",
    license="GNU LGPL v3",
    classifiers=[
        "License :: OSI Approved :: GNU LGPL v3",
        "Programming Language :: Python"
    ],
    packages=setuptools.find_packages(), # type: ignore
    python_requires=">=3.8"
)