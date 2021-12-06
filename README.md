# Awesome Pandas Alternatives [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of awesome Python frameworks, libraries, software and resources.
 
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Pandas Shortcomings](#pandas-shortcomings)
- [Almost Under Every Hood: Apache Arrow](#almost-under-every-hood-apache-arrow)
- [Modern DataFrame Libraries](#modern-dataframe-libraries)
- [Distributed Computing Libraries](#distributed-computing-libraries)
  - [Row-based storage](#row-based-storage)
  - [Columnar Based Storage](#columnar-based-storage)
    - [Using Rust](#using-rust)
    - [Using Ray](#using-ray)
  - [Higher Level APIs](#higher-level-apis)
- [GPU Libraries](#gpu-libraries)
- [Other Libraries and Ports from R](#other-libraries-and-ports-from-r)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Pandas Shortcomings

We all love [`pandas`](https://github.com/pandas-dev/pandas) and most of us have learnt how to do data manipulation using this amazing library. However useful and great `pandas` is, unfortunately it has [some well-known shortcomings](https://youtu.be/ZTXFQ2sEarQ?t=1642) that developers have been trying to address in the past years. Here are the most common weak points:

* You might read that `pandas` generall requires as much RAM as 5-10 times the dataset you are working with, mostly because many operations generate under the hood a in-memory copy of the data;
* `pandas` eagerly executes code, i.e. executes every statement sequentially. For example, if you read a `.csv` file and then filter it on a specific column (say, `col2 > 5`), the whole dataset will be first read into memory and then the subset you are interested in will be returned. Of course, you could manually write `pandas` command sequentially to improve on efficiency, but one can only do so much. For this reason, many of the pandas alternatives implement `lazy` evaluation - i.e. do not execute statements until a `.collect()` or `.compute()` method is called - and include a query execution engine to **optimise the order of the operations** (read more [here](https://duckdb.org/2021/05/14/sql-on-pandas.html)).

This awesome-repo aims to gather the libraries meant to overcome `pandas` weaknesses, as well as some resources to learn them. Everyone is encouraged to add new projects, or edit the descriptions to correct potential mistakes.

## Almost Under Every Hood: Apache Arrow

Most of these libraries leverage [Apache Arrow](https://arrow.apache.org/), "a language-independent columnar memory format". In other words, unlike good old  `.csv` files that are stored by rows, Apache Arrow storage is (also) column-based. This allows partitioning the data in chunks with a lot of [clever tricks](https://arrow.apache.org/docs/format/Columnar.html) to enable greated compression (like storing sequences of repeated values) and faster queries (because each chunk also stores metadata like the min or max value).

Arrow offers Python bindings with its Python API, named [`pyarrow`](https://arrow.apache.org/docs/python/). This library has modules to [read and write data](https://arrow.apache.org/cookbook/py/io.html) with either Arrow formats (`.parquet` most notably, but also `.feather`) and other formats like `.csv` and `.json`, but also data from cloud storage services and in a streaming fashion, which means data is processed in batches and does not need to be read wholly into memory. The module `pyarrow.compute` allows to perform [basic data manipulation](https://arrow.apache.org/cookbook/py/data.html).

On its own, `pyarrow` is rarely being used as a standalone library to perform data manipulation: usually more expressive and feature rich modules are built upon Arrow, especially on its fast C++ or Rust API interface. For this reason, most of the libraries listed here will display a general landing page and links to other languages APIs (mostly, Python and R). To be honest, the R `{arrow}` interface has a backend for [`{dplyr}`](https://github.com/tidyverse/dplyr) (the equivalent of `pandas` in R), which makes its use more straightforward. Development for the R `{arrow}` package is [quite active](https://arrow.apache.org/blog/2021/11/08/r-6.0.0/)!

## Modern DataFrame Libraries

These libraries leverage Apache Arrow memory format to implement a parallelised and lazy execution engine. These are designed to take advantage of all the cores (and threads) of a machine.

* [`polars`](https://github.com/pola-rs/polars): Polars claims to be "a blazingly fast DataFrames library implemented in Rust using Apache Arrow Columnar Format as memory model". It leverages [(quasi-)lazy evaluation](https://pola-rs.github.io/polars-book/user-guide/index.html#introduction), uses all cores, has multithreaded operations and its query engine is written in Rust. `polars` has an expressive and high-level API that enables complex operations.
* [`duckdb`](https://github.com/duckdb/duckdb): is another fairly recent DataFrame library. It offers both a SQL interface and a Python API: in other words, it can be used to query `.csv` and Arrow's `.parquet` files, but also in-memory `pandas.DataFrame`s using both SQL and a syntax closer to Python or `pandas`. It supports window functions.
  * `duckdb` has a very nice blog that explains its optimistations under the hood: for example, [here](https://duckdb.org/2021/06/25/querying-parquet.html) you can find an overview of how Apache's `.parquet` format works and the performance tricks used by the query engine to run **several orders of magnitude faster** than `pandas`.

## Distributed Computing Libraries

Compared to the libraries above, the following are meant to orchestrate data manipulation over clusters, i.e. distribute computing across several nodes (multiple machines with multiple processors) via a query execution engine. Since they need to plan execution preemptively, they can be slower than `pandas` on a single-core machine.

### Row-based storage

These are the "first generation" of query planners that - as of now - are not built around columnar storage. Nonetheless, they represent the industry standard for distributed computing, and offer much more than data manipulation: they can even implement machine learning libraries to train models on the cluster. The downside is that moving from `pandas` to, say, Apache's `spark` is not straightforward, as the API syntax can differ.

* [`dask`](https://github.com/dask/dask) is among the first distributed computing DataFrame libraries, alongside `spark`. `dask` does not only offer a [distributed computing equivalent](https://dask.org/) of `pandas`, but also of [`numpy`](https://github.com/numpy/numpy) and [`scikit-learn`](https://github.com/scikit-learn/scikit-learn), so that "you don't have to completely rewrite your code or retrain to scale up", i.e. to run code on distributed clusters. A Dask DataFrame "is a large parallel DataFrame composed of many smaller Pandas DataFrames, split along the index".
* [`spark`](https://github.com/apache/spark): the de-facto leader of distributed computing, now it is faring more competition. It offers [`MLib`](https://spark.apache.org/mllib/), a machine learning library, as well as [`GraphX`](https://spark.apache.org/graphx/) for "graphs and graph-parallel computation". Its Python API is [`pyspark`](https://github.com/apache/spark/tree/master/python).

### Columnar Based Storage

The "next generation" of distributed query planners, that leverage columnar-based storage.

#### Using Rust

* [`arrow-datafusion`](https://github.com/apache/arrow-datafusion): this is [Arrow's query engine](https://arrow.apache.org/datafusion/user-guide/introduction.html), written in the Rust programming language. Much like `duckdb`, `datafusion` offers both a SQL and Python-like API interface. Much like `pyspark`, `datafusion` "allows you to build a plan through SQL or a DataFrame API against in-memory data, parquet or CSV files, run it in a multi-threaded environment, and obtain the result back in Python". 
  * [`ballista`](https://github.com/apache/arrow-datafusion/blob/master/ballista/README.md) is the distributed query engine, written in Rust, built on top of `arrow-datafusion`. [Here](https://arrow.apache.org/datafusion/user-guide/distributed/introduction.html#how-does-this-compare-to-apache-spark) is a comparison between `ballista` and `spark`.
    * `ballista` [will offer](https://arrow.apache.org/datafusion/user-guide/distributed/clients/python.html) a Python client API.

#### Using Ray

The following libraries use [Ray](https://www.ray.io/) as their default distributing engine. Ray is a Python API for building distributed applications (or, in technical jargon, "scaling your code") - i.e., it offers a series of functions to make your code run on multiple computing nodes. The community has built [many libraries](https://docs.ray.io/en/latest/ray-libraries.html) to build a bridge between Ray and libraries such as XGBoost and scikit-learn, lightGBM, PyTorch Lightning, Ariflow, PyCaret and many more. Ray has a Dataset class to load and exchange data in Ray libraries and applications. As explained in the documentation, "Datasets provide basic distributed data transformations such as map, filter, and repartition, and are compatible with a variety of file formats, datasources, and distributed frameworks", i.e. can be used to preprocess data for machine learning models but also for other distributed frameworks such as `dask` and `pyspark`.

* [`modin`](https://github.com/modin-project/modin) attempts to parallelize as much of the `pandas` API as is possible. Its developers claim to "have worked through a significant portion of the DataFrame API" such much so that `modin` "is intended to be used as a drop-in replacement for `pandas`, such that even if the API is not yet parallelized, it is still defaulting to `pandas`". The library is currently under active development.
  * Under the hood, `modin` calls either to `dask`, `ray` or [OmiSciDB](https://www.omnisci.com/platform/omniscidb) to automatically orchestrate the tasks across cores on your machine or cluster. A good overview of what happens is [here](https://medium.com/distributed-computing-with-ray/how-to-speed-up-pandas-with-modin-84aa6a87bcdb).
  * Offers a spreadsheet-like API to modify dataframes and a whole lot of other [experimental features](https://modin.readthedocs.io/en/stable/experimental_features/index.html#), like using SQL queries on DataFrames and even fitting XGBoost models using distributed computing.
  * Has excellent documentation, including explanatory articles on the differences between `modin` and [`pandas`](https://modin.readthedocs.io/en/stable/comparisons/pandas.html) or [`dask`](https://github.com/modin-project/modin/blob/master/docs/modin_vs_dask.md) (also [here](https://modin.readthedocs.io/en/stable/comparisons/dask.html)).
* [`mars`](https://github.com/mars-project/mars) is a "unified framework for large-scale data computation", like `dask`, but is *tensor-based*. It offers modules to scale numpy, pandas, scikit-learn and many other libraries. 

### Higher Level APIs

These are libraries that implement an abstract layer to make `pandas` code easier to reuse across distributed frameworks, mainly `dask` and `spark`. It is to be noted that, [on October 2021](https://databricks.com/blog/2021/10/04/pandas-api-on-upcoming-apache-spark-3-2.html), `pyspark` adopted a `pandas`-like API.

* [`fugue`](https://github.com/fugue-project/fugue) is "a unified interface for distributed computing that lets users execute Python, pandas, and SQL code on Spark and Dask without rewrites". Simply put, `fugue` adds an abstract layer that makes code portable between across differing computing frameworks.
* [`koalas`](https://github.com/databricks/koalas) is a wrapper around `spark` that offers `pandas`-like APIs. It is no longer developed, since `pyspark` adopted the `pandas` API and `koalas` was merged into `pyspark`. 

## GPU Libraries

Generally libraries work on CPUs and clusters are usually made up of CPUs. Apart from some notable exceptions, such as deep learning libraries like Tensorflow and PyTorch, usually regular libraries do not work on GPUs. This is due to major architectural differences across the two chips.

* [`cuDF`](https://github.com/rapidsai/cudf) is a GPU dataframe library, which is part of the [RapidsAI framework](https://rapids.ai/), that enables "end-to-end data science and analytics pipelines entirely on GPUs". There are many other libraries, like `cuML` for machine learning, `cuspatial` for spatial data manipulations, and more. `cuDF` is based on Apache Arrow, because the memory format is compatible with both CPU and GPU architecture.
* [`blazingSQL`](https://github.com/BlazingDB/blazingsql) is "is a GPU accelerated [distributed] SQL engine built on top of the RAPIDS ecosystem" and, as such, leverages Apache Arrow. Think of this as Apache `spark` on GPU.

## Other Libraries and Ports from R

R has an amazing library called `{dplyr}` that enables easy data manipulation. `{dplyr}` is part of the so-called [`{tidyverse}`](https://www.tidyverse.org/), a unified framework for data manipulation and visualisation.

* [`pyjanitor`](https://github.com/pyjanitor-devs/pyjanitor) was originally conceived as a `pandas` extension of the well-known `{janitor}` R package. The latter was a package to clean strings with ease in a R'`data.frame` or `tibble` objects, but later incorporated new methods to make it more similar to `{dplyr}`. Adding and removing column, for example, is easier with the dedicated methods `df.remove_column()` and `df.add_column()`, but also renaming column is easier with `df.rename_column()`. This enables to run smoother pipelines that exploit *method chaining*.
* [`pydatatable`](https://github.com/h2oai/datatable) is a Python port of the astounding [`{data.table}`](https://github.com/Rdatatable/data.table) library in R, that achieves impressive results thanks to parallelisation.
* [`pandasql`](https://github.com/yhat/pandasql/) allows to query `pandas.DataFrame`s using SQL syntax.
