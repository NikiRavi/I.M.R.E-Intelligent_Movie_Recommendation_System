# import statements
import os
import time
import gc
import sys

# libraries required for Recommender
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType, StringType

# class to implement Alternating Least Squares Approach to recommend movies
class Recommender:
    """
    Collabarative Filter using Alternating Least Square-Model based(ALS-WS)
    Matrix Factorization approach- implemented by MLib-Spark
    """
    # function to define the model and its associated variables
    def __init__(self, spark_session, path_movies, path_ratings):
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.moviesDF = self._load_file(path_movies) \
                            .select(['movieId', 'title', 'genres'])
        self.ratingsDF = self._load_file(path_ratings) \
                             .select(['userId', 'movieId', 'rating'])
        self.model = ALS(
            userCol='userId',
            itemCol='movieId',
            ratingCol='rating',
            coldStartStrategy="drop") # dropping coldStartStrategy let's us use only the non-NaN fields for evaluating the model

    def _load_file(self, filepath):
        """
        read data from file.csv
        """
        return self.spark.read.load(filepath, format='csv', header=True, inferSchema=True)

    def tune_model(self, maxIter, regParams, ranks, split_ratio=(0.6,0.2,0.2)):
        """
        Hyperparameter tuning for ALS model

        Arguements
        ----------
        maxIter: int, number of iterations
        regParams: list of float, regularization parameter
        ranks: list of float, number of latent/hidden factors
        split_ratio: tuple, (train, validation, test)
        """
        # split data
        train, val, test = self.ratingsDF.randomSplit(split_ratio)
        train.cache()
        val.cache()
        test.cache()
        # validation- tuning
        self.model = tune_ALS(self.model, train, val,maxIter, regParams, ranks)
        # test model
        predictions = self.model.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse",
                                        labelCol="rating",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print('The out-of-sample RMSE of the best tuned model is:', rmse)
        #save the best model
        self.model.write().overwrite().save("als_model")
        # clean up
        del train, val, test, predictions, evaluator
        gc.collect()

    def set_model_params(self, maxIter, regParam, rank):
        """
        set model params for pyspark.ml.recommendation.ALS

        Arguements
        ----------
        maxIter: int, max number of learning iterations
        regParams: float, regularization parameter
        ranks: int, number of latent/hidden factors
        """

        self.model = self.model \
            .setMaxIter(maxIter) \
            .setRank(rank) \
            .setRegParam(regParam)

    def _regex_matching(self, fav_movie):
        """
        return the closest matches via SQL regex.
        If no match found, return None

        Arguements
        ----------
        fav_movie: str, name of input movie by the user

        Return
        ------
        list of indices of the matching movies
        """

        print('You have input movie:', fav_movie)
        matchesDF = self.moviesDF \
            .filter(lower(col('title')).like('%{}%'.format(fav_movie.lower()))) \
            .select('movieId', 'title')
        if not len(matchesDF.take(1)):
            print('Arr, Sorry! No match found: Please key in a more complete movie name')
        else:
            movieIds = matchesDF.rdd.map(lambda r: r[0]).collect()
            titles = matchesDF.rdd.map(lambda r: r[1]).collect()
            print('Found possible matches:'
                  '{0}\n'.format([x for x in titles]))
            return movieIds

    def _append_ratings(self, userId, movieIds):
        """
        append a user's movie ratings to ratingsDF (Implicit feedback- Considering view as like)

        Arguements
        ---------
        userId: int, userId of a user
        movieIds: int, movieIds of user's favorite movies
        """
        # create new user rdd
        user_rdd = self.sc.parallelize(
            [(userId, movieId, 5.0) for movieId in movieIds])
        # transform to user rows
        user_rows = user_rdd.map(
            lambda x: Row(userId=int(x[0]),movieId=int(x[1]),rating=float(x[2])))
        # transform rows to spark DF
        userDF = self.spark.createDataFrame(user_rows) \
            .select(self.ratingsDF.columns)
        # append to ratingsDF
        self.ratingsDF = self.ratingsDF.union(userDF)

    def _create_inference_data(self, userId, movieIds):
        """
        create DF with all movies except ones that were rated for inferencing (the movie(s) that match the user's search)
        """
        # filter movies
        other_movieIds = self.moviesDF \
            .filter(~col('movieId').isin(movieIds)) \
            .select(['movieId']) \
            .rdd.map(lambda r: r[0]) \
            .collect()
        # create inference rdd
        inferenceRDD = self.sc.parallelize([(userId, movieId) for movieId in other_movieIds]).map(lambda x: Row(userId=int(x[0]),movieId=int(x[1]),))
        # transform to inference DF
        inferenceDF = self.spark.createDataFrame(inferenceRDD) \
            .select(['userId', 'movieId'])
        return inferenceDF

    def _inference(self, model, fav_movie, n_recommendations=10):
        """
        return top n movie recommendations based on user's input movie

        Arguements
        ----------
        model: spark ALS model
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations: by default 10

        Return
        ------
        list of top n similar movie recommendations
        """
        # create a userId
        userId = self.ratingsDF.agg({"userId": "max"}).collect()[0][0] + 1
        # get movieIds of favorite movies
        if fav_movie:
            movieIds = self._regex_matching(fav_movie)
            if movieIds != None:
                # append new user with his/her ratings into data
                self._append_ratings(userId, movieIds)
        else:
            movieIds = ""
        # matrix factorization
        model = model.fit(self.ratingsDF)
        # get data for inferencing
        inferenceDF = self._create_inference_data(userId, movieIds)
        # make inference
        return model.transform(inferenceDF) \
            .select(['movieId', 'prediction']) \
            .orderBy('prediction', ascending=False) \
            .rdd.map(lambda r: (r[0], r[1])) \
            .take(n_recommendations*n_recommendations)

    def initial_recommendation(self, movie_name):
        """
        make top n movie recommendations for the first time before the system gets to judge the user

        Arguements
        ----------
        n_recommendations: int, top n recommendations
        """
        # make inference and get raw recommendations
        print('Recommendation system start to make inference ...')
        t0 = time.time()
        raw_recommends = \
        self._inference(self.model, movie_name, 10)
        movieIds = [r[0] for r in raw_recommends]
        scores = [r[1] for r in raw_recommends]
        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))
        # recommendation based on genre- Action
        movie_action = self.moviesDF \
            .filter(col('movieId').isin(movieIds)) \
            .where("genres like '%Action%'") \
            .select('title') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        movie_action = movie_action[:10]
        # recommendation based on genre- comedy
        movie_comedy = self.moviesDF \
            .filter(col('movieId').isin(movieIds)) \
            .where("genres like '%Comedy%'") \
            .select('title') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        movie_comedy = movie_comedy[:10]
        # recommendation based on genre- Thriller
        movie_thriller = self.moviesDF \
            .filter(col('movieId').isin(movieIds)) \
            .where("genres like '%Thriller%'") \
            .select('title') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        movie_thriller = movie_thriller[:10]
        print('thriller', movie_thriller)
        # recommendation based on genre- Romance
        movie_romance = self.moviesDF \
            .filter(col('movieId').isin(movieIds)) \
            .where("genres like '%Romance%'") \
            .select('title') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        movie_romance = movie_romance[:10]
        # recommendation based on genre- Sci-fi
        movie_sci = self.moviesDF \
            .filter(col('movieId').isin(movieIds)) \
            .where("genres like '%Sci-Fi%'") \
            .select('title') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        movie_sci = movie_sci[:10]
        # recommendation based on genre- Drama
        movie_drama = self.moviesDF \
            .filter(col('movieId').isin(movieIds)) \
            .where("genres like '%Drama%'") \
            .select('title') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        movie_drama = movie_drama[:10]
        # recommendation based on genre- Horror
        movie_horror = self.moviesDF \
            .filter(col('movieId').isin(movieIds)) \
            .where("genres like '%Horror%'") \
            .select('title') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        movie_horror = movie_horror[:10]
        # recommendation based on genre- Children
        movie_kid = self.moviesDF \
            .filter(col('movieId').isin(movieIds)) \
            .where("genres like '%Children%'") \
            .select('title') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        movie_kid = movie_kid[:10]
        print('Recommendations based on genre: Romance')
        for j in range(len(movie_romance)):
            print('{0}: {1}'
                  'of'.format(j+1, movie_romance[j]))
        print('\nRecommendations based on genre: Action')
        for j in range(len(movie_action)):
            print('{0}: {1}'
                  'of'.format(j+1, movie_action[j]))
        print('\nRecommendations based on genre: Children')
        for j in range(len(movie_kid)):
            print('{0}: {1}'
                  'of'.format(j+1, movie_kid[j]))
        print('\nRecommendations based on genre: Thriller')
        for j in range(len(movie_thriller)):
            print('{0}: {1}'
                  'of'.format(j+1, movie_thriller[j]))
        print('\nRecommendations based on genre: Horror')
        for j in range(len(movie_horror)):
            print('{0}: {1}'
                  'of'.format(j+1, movie_horror[j]))
        print('\nRecommendations based on genre: Science Fiction')
        for j in range(len(movie_sci)):
            print('{0}: {1}'
                  'of'.format(j+1, movie_sci[j]))
        print('\nRecommendations based on genre: Comedy')
        for j in range(len(movie_comedy)):
            print('{0}: {1}'
                  'of'.format(j+1, movie_comedy[j]))
        print('\nRecommendations based on genre: Drama')
        for j in range(len(movie_drama)):
            print('{0}: {1}'
                  'of'.format(j+1, movie_drama[j]))
        comedy = spark.createDataFrame(movie_comedy, StringType())
        romance = spark.createDataFrame(movie_romance, StringType())
        drama = spark.createDataFrame(movie_drama, StringType())
        horror = spark.createDataFrame(movie_horror, StringType())
        children = spark.createDataFrame(movie_kid, StringType())
        sci_fi = spark.createDataFrame(movie_sci, StringType())
        thriller = spark.createDataFrame(movie_thriller, StringType())
        action = spark.createDataFrame(movie_action, StringType())

    def make_recommendations(self, fav_movie):
        """
        make top n movie recommendations

        Parameters
        ----------
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations
        """
        # make inference and get raw recommendations
        print('Recommendation system start to make inference ...')
        t0 = time.time()
        raw_recommends = \
            self._inference(self.model, fav_movie, 10)
        movieIds = [r[0] for r in raw_recommends]
        scores = [r[1] for r in raw_recommends]
        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))

        # get movie titles
        movie_ratings = self.moviesDF \
            .filter(col('movieId').isin(movieIds)) \
            .select('title') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        movie_ratings = movie_ratings[:10]

        # print recommendations
        print('Recommendations for {}:'.format(fav_movie))
        for i in range(len(movie_ratings)):
            print('{0}: {1}, with rating '
                  'of {2}'.format(i+1, movie_ratings[i], scores[i]))
        recommended_based_input = spark.createDataFrame(movie_ratings, StringType())


def tune_ALS(model, train_data, validation_data, maxIter, regParams, ranks):#validation_data,
    """
    function to select the best model based on RMSE of validation data

    Arguements
    ----------
    model: spark ML model, ALS
    train_data: spark DF with columns ['userId', 'movieId', 'rating']
    validation_data: spark DF with columns ['userId', 'movieId', 'rating']
    maxIter: int, max number of iterations to learn Parameters
    regParams: list of float, one dimension of hyper-param tuning grid
    ranks: list of float, one dimension of hyper-param tuning grid

    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initialize
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # get ALS model
            als = ALS(maxIter=maxIter, regParam=reg, rank=rank, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
            # train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print('{} latent/hidden factors and regularization = {}: '
                  'RMSE on validation data is {}'.format(rank, reg, rmse))
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and '
          'regularization = {}'.format(best_rank, best_regularization))
    return best_model

if __name__ == '__main__':
    # get args
    movies_filename = sys.argv[1]
    ratings_filename = sys.argv[2]
    movie_name = sys.argv[3]
    # initialize spark instance
    spark = SparkSession \
        .builder \
        .appName("movie recommender") \
        .getOrCreate()
    # initialiaze our recommender
    recommender = Recommender(spark, movies_filename, ratings_filename)
    # initialize Hyperparameters for ALS
    recommender.set_model_params(10, 0.01, 50)
    #tune model- already tuned!! hence commented
    #recommender.tune_model(5, [0.001, 0.001, 0.5], ranks=[3,8,12,20,50], split_ratio=(0.6,0.2,0.2))
    #initial predictions
    recommender.initial_recommendation(movie_name)
    # make recommendations
    recommender.make_recommendations(movie_name)
    # stop spark instance
    spark.stop()
