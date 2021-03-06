import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession, functions, types,Row
from pyspark.sql.functions import sum, count, udf
import math

import re
spark = SparkSession.builder.appName('imre').getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext (sc)
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert sc.version >= '2.3'  # make sure we have Spark 2.3+

def colname(col):
    year = None
    if col[-1:] == ')':
        year = col[-5:-1]
    return year

def col(col):
    movie_title = None
    if col[-1:] == ')':
        movie_title = col[:-6]
    return movie_title

def main(keyspace):

    ratings_schema = types.StructType([
    types.StructField('userid', types.IntegerType(), False),
    types.StructField('movieid', types.IntegerType(), False),
    types.StructField('rating', types.FloatType(), False)
    ])

    movies_schema = types.StructType([
    types.StructField('movieid', types.IntegerType(), False),
    types.StructField('title', types.StringType(), False),
    types.StructField('genres', types.StringType(), False)
    ])

    scores_schema = types.StructType([
    types.StructField('movieid', types.IntegerType(), False),
    types.StructField('tagid', types.IntegerType(), False),
    types.StructField('relevance', types.FloatType(), False)
    ])

    tags_schema = types.StructType([
    types.StructField('tagid', types.IntegerType(), False),
    types.StructField('tag', types.StringType(), False)
    ])

    links_schema = types.StructType([
    types.StructField('movieid', types.IntegerType(), False),
    types.StructField('imdbid', types.IntegerType(), False),
    types.StructField('tmdbid', types.IntegerType(), False)
    ])

    movie_schema = types.StructType([
    types.StructField('id', types.IntegerType(), False),
    types.StructField('language', types.StringType(), False),
    types.StructField('plot', types.StringType(), False),
    types.StructField('popularity', types.FloatType(), False),
    types.StructField('poster', types.StringType(), False),
    types.StructField('production_companies', types.StringType(), False),
    types.StructField('release_date', types.DateType(), False),
    types.StructField('keywords', types.StringType(), False),
    types.StructField('vote_average', types.FloatType(), False)
    ])

    # Clean, Aggregate and Load data into the table imre_movies_repo with the movie records with the other required features

    ratings = spark.read.csv ("ratings.csv", schema = ratings_schema).createOrReplaceTempView("ratings")
    movies = spark.read.csv ("movies.csv", schema = movies_schema).createOrReplaceTempView("movies")
    movie_ratings = spark.sql("SELECT userid, ratings.movieid, rating, title, genres From ratings JOIN movies ON ratings.movieid = movies.movieid ").createOrReplaceTempView("movie_ratings")
    links = spark.read.csv("links.csv", schema = links_schema).createOrReplaceTempView("links")
    link_table = spark.sql("SELECT userid, movie_ratings.movieid, rating, title, genres, tmdbid From movie_ratings JOIN links ON movie_ratings.movieid = links.movieid").createOrReplaceTempView("link_table")
    data = spark.read.csv ("moviedata.csv", schema = movie_schema).createOrReplaceTempView("data")
    data_table = spark.sql("SELECT userid, movieid, rating, title as titleyear, genres, link_table.tmdbid, language, plot, popularity, poster, production_companies, keywords, vote_average from link_table JOIN data ON link_table.tmdbid = data.id")

    udf_split_year = udf(colname, types.StringType())
    udf_split_title = udf(col, types.StringType())
    split_table = data_table.withColumn("release_year", udf_split_year("titleyear")).withColumn("title", udf_split_title("titleyear")).drop("titleyear").createOrReplaceTempView("split_table")
    final_table = spark.sql("SELECT userid, movieid, genres, keywords, language, plot, popularity, poster, production_companies, rating, release_year, title, tmdbid, vote_average from split_table")
    final_table.show()

    # Write the data into the Casssandra Table
    final_table.write.format("org.apache.spark.sql.cassandra").options(table='imre_movies_repo', keyspace=keyspace).save()

    # Clean, Aggregate and Load data into the table imre_tags with the movie tags and relevance information

    genome_scores = spark.read.csv ("genome-scores.csv", schema = scores_schema).createOrReplaceTempView("gscores")
    genome_tags = spark.read.csv ("genome-tags.csv", schema = tags_schema).createOrReplaceTempView("gtags")
    scores_tags = spark.sql("SELECT movieid, gscores.tagid, relevance, tag From gscores JOIN gtags ON gscores.tagid = gtags.tagid group by movieid, gscores.tagid, relevance, tag order by movieid, tagid")
    groupeddata = scores_tags.groupby("movieid").agg(functions.collect_list("tagid"),functions.collect_list("relevance"),functions.collect_list("tag"))
    renamedgroup = groupeddata.withColumnRenamed("collect_list(tagid)", "tagid").withColumnRenamed("collect_list(relevance)", "relevance").withColumnRenamed("collect_list(tag)", "tag").createOrReplaceTempView("renamedgroup")
    relevance_table = spark.sql("SELECT movieid, relevance, tag, tagid from renamedgroup")
    relevance_table.show()

    # Write the data into the Casssandra Table
    relevance_table.write.format("org.apache.spark.sql.cassandra").options(table='imre_tags', keyspace=keyspace).save()

if __name__ == '__main__':
    keyspace = sys.argv[1]
    cluster_seeds = ['199.60.17.188', '199.60.17.216']
    spark = SparkSession.builder.appName('Spark Cassandra example').config('spark.cassandra.connection.host', ','.join(cluster_seeds)).config('spark.dynamicAllocation.maxExecutors', 16).getOrCreate()
    main(keyspace)
