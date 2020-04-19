# import statements
import os
import time
import gc
import sys

# libraries required for Recommender
from pyspark.sql import SparkSession, Row
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType, StringType

def load_file(spark_session, filepath):
    """
    read data from file.csv
    """
    return spark_session.read.load(filepath, format='csv', header=True, inferSchema=True)

def main(spark_session, data):
    spark = spark_session
    sc = spark_session.sparkContext
    sqlContext = SQLContext(sc)
    datasetDF = load_file(spark, data) \
                        .select(['movieid', 'userid', 'genres', 'language', 'plot', 'popularity', 'poster', 'production_companies', 'rating', 'release_year', 'title', 'vote_average', 'tmdbid'])
    datasetDF.show(20)
    datasetDF.cache()
    datasetDF_production_companies = datasetDF.groupBy('production_companies').agg(mean(datasetDF.rating).alias('average rating'), mean(datasetDF.vote_average).alias('average vote'), mean(datasetDF.popularity).alias('average popularity'), max(datasetDF.release_year).alias('release year'), max(datasetDF.userid).alias('userid'))
    datasetDF_production_companies.cache()
    datasetDF_production_companies = datasetDF_production_companies.filter(datasetDF_production_companies['production_companies'].isin(datasetDF['production_companies']))
    datasetDF_production_companies = datasetDF_production_companies.filter(datasetDF_production_companies['average rating'].isNotNull()).filter(datasetDF_production_companies['average vote'].isNotNull()).filter(datasetDF_production_companies['average popularity'].isNotNull()).filter(datasetDF_production_companies['release year'].isNotNull()).filter(datasetDF_production_companies['userid'].isNotNull())
    datasetDF_production_companies.show(50)
    datasetDF_production_companies.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("production")
    # setting output path
    output_path = 'Plotlyjs/production/'
    #creating system command line
    cmd_production = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'production.csv'
    #executing system command
    os.system(cmd_production)
    datasetDF_movies = datasetDF.groupBy('title').agg(mean(datasetDF.rating).alias('average rating'), mean(datasetDF.vote_average).alias('average vote'), mean(datasetDF.popularity).alias('average popularity'), max(datasetDF.release_year).alias('release year'))
    datasetDF_movies.cache()
    datasetDF_movies = datasetDF_movies.filter(datasetDF_movies['title'].isin(datasetDF['title']))
    datasetDF_movies = datasetDF_movies.filter(datasetDF_movies['average vote'].isNotNull()).filter(datasetDF_movies['average rating'].isNotNull()).filter(datasetDF_movies['average popularity'].isNotNull()).filter(datasetDF_movies['release year'].isNotNull())
    datasetDF_movies.show(50)
    datasetDF_movies.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("movies")
    # setting output path
    output_path = 'Plotlyjs/movies/'
    #creating system command line
    cmd_movies = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'movies_final.csv'
    #executing system command
    os.system(cmd_movies)
    datasetDF_users = datasetDF.groupBy('userid').agg(mean(datasetDF.rating).alias('average rating'), mean(datasetDF.vote_average).alias('average vote'), mean(datasetDF.popularity).alias('average popularity'), max(datasetDF.release_year).alias('release year'))
    datasetDF_users.cache()
    datasetDF_users = datasetDF_users.filter(datasetDF_users['userid'].isin(datasetDF['userid']))
    datasetDF_users = datasetDF_users.filter(datasetDF_users['average rating'].isNotNull()).filter(datasetDF_users['average vote'].isNotNull()).filter(datasetDF_users['average popularity'].isNotNull()).filter(datasetDF_users['release year'].isNotNull())
    datasetDF_users.show(50)
    datasetDF_users.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("users")
    # setting output path
    output_path = 'Plotlyjs/users/'
    #creating system command line
    cmd_users = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'users.csv'
    #executing system command
    os.system(cmd_users)
    datasetDF_genre = datasetDF.groupBy('genres').agg(count(datasetDF.title).alias('title'))
    datasetDF_genre.cache()
    datasetDF_genre = datasetDF_genre.filter(datasetDF_genre['genres'].isin(datasetDF['genres']))
    datasetDF_genre = datasetDF_genre.filter(datasetDF_genre['title'].isNotNull())
    datasetDF_genre.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("genre")
    # setting output path
    output_path = 'Plotlyjs/genre/'
    #creating system command line
    cmd_genre = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'genre-movie.csv'
    #executing system command
    os.system(cmd_genre)
    datasetDF_lang = datasetDF.groupBy('language').agg(count(datasetDF.title).alias('title'))
    datasetDF_lang.cache()
    datasetDF_lang = datasetDF_lang.filter(datasetDF_lang['language'].isin(datasetDF['language']))
    datasetDF_lang = datasetDF_lang.filter(datasetDF_lang['title'].isNotNull())
    datasetDF_lang.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("lang")
    # setting output path
    output_path = 'Plotlyjs/lang/'
    #creating system command line
    cmd_lang = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'lang-movie.csv'
    #executing system command
    os.system(cmd_lang)
    datasetDF_vote = datasetDF.groupBy('vote_average').agg(count(datasetDF.title).alias('title'))
    datasetDF_vote.cache()
    datasetDF_vote = datasetDF_vote.filter(datasetDF_vote['vote_average'].isin(datasetDF['vote_average']))
    datasetDF_vote = datasetDF_vote.filter(datasetDF_vote['title'].isNotNull())
    datasetDF_vote.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("vote")
    # setting output path
    output_path = 'Plotlyjs/vote/'
    #creating system command line
    cmd_vote = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'vote-movie.csv'
    #executing system command
    os.system(cmd_vote)
    datasetDF_lpro = datasetDF.groupBy('production_companies').agg(count(datasetDF.title).alias('title'))
    datasetDF_lpro.cache()
    datasetDF_lpro = datasetDF_lpro.filter(datasetDF_lpro['production_companies'].isin(datasetDF['production_companies']))
    datasetDF_lpro = datasetDF_lpro.filter(datasetDF_lpro['title'].isNotNull())
    datasetDF_lpro.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("production-mov")
    # setting output path
    output_path = 'Plotlyjs/production-mov/'
    #creating system command line
    cmd_lpro = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'production-movie.csv'
    #executing system command
    os.system(cmd_lpro)
    datasetDF_langpro = datasetDF.groupBy('production_companies').agg(count(datasetDF.language).alias('language'))
    datasetDF_langpro.cache()
    datasetDF_langpro = datasetDF_langpro.filter(datasetDF_langpro['production_companies'].isin(datasetDF['production_companies']))
    datasetDF_langpro = datasetDF_langpro.filter(datasetDF_langpro['language'].isNotNull())
    datasetDF_langpro.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("lang-pro")
    # setting output path
    output_path = 'Plotlyjs/lang-pro/'
    #creating system command line
    cmd_lanpro = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'lang-pro.csv'
    #executing system command
    os.system(cmd_lanpro)
    datasetDF_ymovie = datasetDF.groupBy('release_year').agg(count(datasetDF.title).alias('title'))
    datasetDF_ymovie.cache()
    datasetDF_ymovie = datasetDF_ymovie.filter(datasetDF_ymovie['release_year'].isin(datasetDF['release_year']))
    datasetDF_ymovie = datasetDF_ymovie.filter(datasetDF_ymovie['title'].isNotNull())
    datasetDF_ymovie.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("year")
    # setting output path
    output_path = 'Plotlyjs/year/'
    #creating system command line
    cmd_ymovie = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'year-movie.csv'
    #executing system command
    os.system(cmd_ymovie)
    datasetDF_ypro = datasetDF.groupBy('release_year').agg(count(datasetDF.production_companies).alias('production_companies'))
    datasetDF_ypro.cache()
    datasetDF_ypro = datasetDF_ypro.filter(datasetDF_ypro['release_year'].isin(datasetDF['release_year']))
    datasetDF_ypro = datasetDF_ypro.filter(datasetDF_ypro['production_companies'].isNotNull())
    datasetDF_ypro.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("year-pro")
    # setting output path
    output_path = 'Plotlyjs/year-pro/'
    #creating system command line
    cmd_ypro = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'year-prod.csv'
    #executing system command
    os.system(cmd_ypro)



if __name__ == '__main__':
    # get args
    dataset_filename = sys.argv[1]
    # initialize spark instance
    spark = SparkSession \
        .builder \
        .appName("analytics") \
        .getOrCreate()

    main(spark, dataset_filename)
    # stop spark instance
    spark.stop()
