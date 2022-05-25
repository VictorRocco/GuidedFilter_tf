import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GuidedFilters",
    version="0.1.0",
    description="Guided Filters layers for TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VictorRocco/GuidedFilters-tf",
    author="Victor Rocco",
    author_email="victor_rocco@hotmail.com",
    # Exclude the build files.
    packages=setuptools.find_packages(exclude=["test"]),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
    zip_safe=True
)

