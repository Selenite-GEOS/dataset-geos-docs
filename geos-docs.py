# TODO: Create venv for python deps and sphinx
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""
This dataset script downloads and builds the GEOS docs dataset.
"""


import json
import os
import datasets

python_deps = [
        #    "h5py",
        #    "scipy",
    "sphinx",
    "matplotlib",
    "sphinx-markdown-builder",
    "sphinx_design",
    "sphinx-argparse",
    "sphinxcontrib-plantuml",
    "sphinxcontrib.programoutput",
    "sphinx-rtd-theme",
]

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "v1": {
        "geos": "https://github.com/GEOS-DEV/GEOS/archive/refs/heads/develop.zip",
    }
    # "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class GeosDocs(datasets.GeneratorBasedBuilder):
    """GeosDocs is a dataset made from the documentation of GEOS."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="v1",
            version=VERSION,
            description="This part of my dataset covers a first domain",
        ),
        # datasets.BuilderConfig(name="second_domain", version=VERSION, description="This part of my dataset covers a second domain"),
    ]

    DEFAULT_CONFIG_NAME = "v1"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if (
            self.config.name == "v1"
        ):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "prompt": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            )
        else:
            raise ValueError(f"unknown config name {self.config.name}")
        # else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option2": datasets.Value("string"),
        #             "second_domain_answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive

        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        import os
        import sys
        import subprocess

        print("Installing python dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *python_deps])

        print("Building docs...")
        geos_src = os.path.join(data_dir["geos"], "GEOS-develop", "src")
        build_dir = os.path.join(data_dir["geos"], "build")
        subprocess.check_call(["sphinx-build", "-M", "markdown", geos_src, build_dir])
        markdown_dir = os.path.join(build_dir, "markdown")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "geos_docs": markdown_dir,
                    "split": "train",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, "dev.jsonl"),
            #         "split": "dev",
            #     },
            # ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, "test.jsonl"),
            #         "split": "test",
            #     },
            # ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, geos_docs, split):
        """ This method handles input defined in _split_generators to yield (key, example) tuples from the dataset. """
        data_files = []
        file_count = 0
        for path, _, files in os.walk(geos_docs):
            for file in files:
                filepath = os.path.join(path, file)
                data_files.append(filepath)

                with open(filepath, encoding="utf-8") as f:
                    data = f.read()
                    if self.config.name == "v1":
                        # Yields examples as (key, example) tuples
                        yield filepath, {
                            "prompt": f"Give me the exact content of the documentation page located at {filepath[len(geos_docs) + 1:]}",
                            "answer": data,
                        }
                    else:
                        raise ValueError(f"Unknown configuration {self.config.name}")
                file_count += 1
                if file_count % 50 == 0:
                    pass
                    # print(f"Parsed {file_count} files")

        print(f"\nParsed {len(data_files)} documentation files.")
