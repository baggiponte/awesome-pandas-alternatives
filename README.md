# Awesome Pandas Alternatives

We all love [`pandas`](https://github.com/pandas-dev/pandas) and most of us have learnt how to do data manipulation using this amazing library. However useful and great `pandas` is, unfortunately it has [some well-known shortcomings](https://youtu.be/ZTXFQ2sEarQ?t=1642) that developers have been trying to address in the past years. Here are the most common weak points:

* You might read that `pandas` generall requires as much RAM as 5-10 times the dataset you are working with, mostly because many operations generate under the hood a in-memory copy of the data;
* `pandas` eagerly executes code, i.e. executes every statement sequentially. For example, if you read a `.csv` file and then filter it on a specific column (say, `col2 > 5`), the whole dataset will be first read into memory and then the subset you are interested in will be returned. Of course, you could manually write `pandas` command sequentially to improve on efficiency, but one can only do so much. For this reason, many of the pandas alternatives implement `lazy` evaluation - i.e. do not execute statements until a `.collect()` or `.compute()` method is called - and include a query execution engine to **optimise the order of the operations** (read more [here](https://duckdb.org/2021/05/14/sql-on-pandas.html)).

This awesome-repo aims to gather the libraries meant to overcome `pandas` weaknesses, as well as some resources to learn them. Everyone is encouraged to add new projects, or edit the descriptions to correct potential mistakes.

## (Almost) Under Every Hood: Apache Arrow

Most of these libraries leverage [Apache Arrow](https://arrow.apache.org/), "a language-independent columnar memory format". In other words, unlike good old  `.csv` files that are stored by rows, Apache Arrow storage is (also) column-based. This allows partitioning the data in chunks with a lot of [clever tricks](https://arrow.apache.org/docs/format/Columnar.html) to enable greated compression (like storing sequences of repeated values) and faster queries (because each chunk also stores metadata like the min or max value).

Arrow offers Python bindings with its Python API, named [`pyarrow`](https://arrow.apache.org/docs/python/). This library has modules to [read and write data](https://arrow.apache.org/cookbook/py/io.html) with either Arrow formats (`.parquet` most notably, but also `.feather`) and other formats like `.csv` and `.json`, but also data from cloud storage services and in a streaming fashion, which means data is processed in batches and does not need to be read wholly into memory. The module `pyarrow.compute` allows to perform [basic data manipulation](https://arrow.apache.org/cookbook/py/data.html).

On its own, `pyarrow` is rarely being used as a standalone library to perform data manipulation: usually more expressive and feature rich modules are built upon Arrow, especially on its fast C++ or Rust API interface. For this reason, most of the libraries listed here will display a general landing page and links to other languages APIs (mostly, Python and R). To be honest, the R `{arrow}` interface has a backend for [`{dplyr}`](https://github.com/tidyverse/dplyr) (the equivalent of `pandas` in R), which makes its use more straightforward. Development for the R `{arrow}` package is [quite active](https://arrow.apache.org/blog/2021/11/08/r-6.0.0/)!

## Multiprocessor/Multithreaded Dataframe Libraries

These libraries leverage Apache Arrow memory format to implement a parallelised and lazy execution engine.

* [`polars`](https://github.com/pola-rs/polars): Polars claims to be "a blazingly fast DataFrames library implemented in Rust using Apache Arrow Columnar Format as memory model". It leverages [(quasi-)lazy evaluation](https://pola-rs.github.io/polars-book/user-guide/index.html#introduction), uses all cores, has multithreaded operations and its query engine is written in Rust. `polars` has an expressive and high-level API that enables complex operations.
* [`duckdb`](https://github.com/duckdb/duckdb): is another fairly recent DataFrame library. It offers both a SQL interface and a Python API: in other words, it can be used to query `.csv` and Arrow's `.parquet` files, but also in-memory `pandas.DataFrame`s using both SQL and a syntax closer to Python or `pandas`. It supports window functions.
  * `duckdb` has a very nice blog that explains its optimistations under the hood: for example, [here](https://duckdb.org/2021/06/25/querying-parquet.html) you can find an overview of how Apache's `.parquet` format works and the performance tricks used by the query engine to run **several orders of magnitude faster** than `pandas`.

## Distributed Computing Libraries

Compared to the libraries above, the following are meant to orchestrate data manipulation over single-node machines or clusters, via a query execution engine. This means that they have a bit of overhead, as they also require to plan in advance how to distribute the workload across, well, workers (i.e., cluster nodes). This is one instance of the so-called [distributed computing](https://www.ibm.com/docs/en/txseries/8.2?topic=overview-what-is-distributed-computing).

Libraries such as `dask` and Apache `spark` come with much more than data manipulation libraries: they can even implement machine learning libraries to train models on the cluster.

* [`dask`](https://github.com/dask/dask) does not only offer a [distributed computing equivalent](https://dask.org/) of `pandas`, but also of [`numpy`](https://github.com/numpy/numpy) and [`scikit-learn`](https://github.com/scikit-learn/scikit-learn), so that "you don't have to completely rewrite your code or retrain to scale up", i.e. to run code on distributed clusters. It currently is not designed around Apache Arrow's columnar memory format.
* [`spark`](https://github.com/apache/spark): the de-facto leader of distributed computing, now it is faring more competition. As of now, it is not designed around Apache Arrow's columnar format. `apache-spark` offers [`MLib`](https://spark.apache.org/mllib/), a machine learning library, as well as [`GraphX`](https://spark.apache.org/graphx/) for "graphs and graph-parallel computation".
* [`fugue-sql`](https://github.com/fugue-project/fugue) is "a unified interface for distributed computing that lets users execute Python, pandas, and SQL code on Spark and Dask without rewrites". In other words, it was designed to have an API closer to that of `pandas` that could be used to re-use code from `pandas` to other distributed dataframe libraries and to simplify the code
* [`arrow-datafusion`](https://github.com/apache/arrow-datafusion): this is [Arrow's query engine](https://arrow.apache.org/datafusion/user-guide/introduction.html), written in Rust. Much like `duckdb`, `datafusion` offers both a SQL and Python-like API interface. Much like `pyspark`, `datafusion` "allows you to build a plan through SQL or a DataFrame API against in-memory data, parquet or CSV files, run it in a multi-threaded environment, and obtain the result back in Python". 
  * [`ballista`](https://github.com/apache/arrow-datafusion/blob/master/ballista/README.md) is the distributed query engine built on top of `arrow-datafusion`.

## GPU Libraries

Generally libraries work on CPUs and clusters are usually made up of CPUs. Apart from some notable exceptions, such as deep learning libraries like Tensorflow and PyTorch, usually regular libraries do not work on GPUs. This is due to major architectural differences across the two chips.

* [`cuDF`](https://github.com/rapidsai/cudf) is a GPU dataframe library, which is part of the [RapidsAI framework](https://rapids.ai/), that enables "end-to-end data science and analytics pipelines entirely on GPUs". There are many other libraries, like `cuML` for machine learning, `cuspatial` for spatial data manipulations, and more. `cuDF` is based on Apache Arrow, because the memory format is compatible with both CPU and GPU architecture.
* [`blazingSQL`](https://github.com/BlazingDB/blazingsql) is "is a GPU accelerated [distributed] SQL engine built on top of the RAPIDS ecosystem" and, as such, leverages Apache Arrow. Think of this as Apache `spark` on GPU.

## Implementations Closer to R Libraries

R has an amazing library called `{dplyr}` that enables easy data manipulation. `{dplyr}` is part of the so-called [`{tidyverse}`](https://www.tidyverse.org/), a unified framework for data manipulation and visualisation.

* [`pyjanitor`](https://github.com/pyjanitor-devs/pyjanitor) was originally conceived as a `pandas` extension of the well-known `{janitor}` R package. The latter was a package to clean strings with ease in a R'`data.frame` or `tibble` objects, but later incorporated new methods to make it more similar to `{dplyr}`. Adding and removing column, for example, is easier with the dedicated methods `df.remove_column()` and `df.add_column()`, but also renaming column is easier with `df.rename_column()`. This enables to run smoother pipelines that exploit *method chaining*.
* [`pydatatable`](https://github.com/h2oai/datatable) is a Python port of the astounding [`{data.table}`](https://github.com/Rdatatable/data.table) library in R, that achieves impressive results thanks to parallelisation.
