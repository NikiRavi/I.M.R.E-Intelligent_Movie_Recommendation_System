# import statements
import os
import time
import gc

# libraries required for AlsRecommender
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, lower
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# class to implement Alternating Least Squares Approach to recommend movies
class AlsRecommender:
    """
    Collabarative Filter using Alternating Least Square-Model based(ALS-WS)
    Matrix Factorization approach- implemented by MLib-Spark (based on Netflix prize paper)
    """
    # function to define the model and its associated variables
    def __init__(self, spark_session, path_movies, path_ratings):
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.moviesDF = self._load_file(path_movies) \
                            .select(['movieId', 'title'])
        self.ratingsDF = self._load_file(path_ratings) \
                             .select(['userId', 'movieId', 'rating']) # change to cassandra
        self.model = ALS(
            userCol='userId',
            itemCol='movieId',
            ratingCol='rating',
            coldStartStrategy="drop") # dropping coldStartStrategy let's us use only the non-NaN fields for evaluating the model

    def _load_file(self, filepath):
        """
        read table from cassandra DB as spark DF
        """
        return self.spark.read.load(filepath, format='csv', header=True, inferSchema=True)

    def model_save(self, spark_session):
        self.save(spark_session, "als_model.model")

    def load_model(self, spark_session):
        return self.load(spark_session, "als_model.model")

    def tune_model(self, maxIter, regParams, ranks, split_ratio=(6, 2, 2)):
        """
        Hyperparameter tuning for ALS model

        Parameters
        ----------
        maxIter: int, max number of learning iterations
        regParams: list of float, regularization parameter
        ranks: list of float, number of latent factors
        split_ratio: tuple, (train, validation, test)
        """
        # split data
        train, val, test = self.ratingsDF.randomSplit(split_ratio)
        # holdout tuning
        self.model = tune_ALS(self.model, train, val, maxIter, regParams, ranks)
        # test model
        predictions = self.model.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse",
                                        labelCol="rating",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print('The out-of-sample RMSE of the best tuned model is:', rmse)
        # clean up
        del train, val, test, predictions, evaluator
        gc.collect()


    def set_model_params(self, maxIter, regParam, rank):
        """
        set model params for pyspark.ml.recommendation.ALS

        Parameters
        ----------
        maxIter: int, max number of learning iterations
        regParams: float, regularization parameter
        ranks: float, number of latent factors
        """

        self.model = self.model \
            .setMaxIter(maxIter) \
            .setRank(rank) \
            .setRegParam(regParam)


    def _regex_matching(self, fav_movie):
        """
        return the closest matches via SQL regex.
        If no match found, return None

        Parameters
        ----------
        fav_movie: str, name of user input movie

        Return
        ------
        list of indices of the matching movies
        """

        print('You have input movie:', fav_movie)
        matchesDF = self.moviesDF \
            .filter(lower(col('title')).like('%{}%'.format(fav_movie.lower()))) \
            .select('movieId', 'title')
        if not len(matchesDF.take(1)):
            print('Arr, Sorry! No match found')
        else:
            movieIds = matchesDF.rdd.map(lambda r: r[0]).collect()
            titles = matchesDF.rdd.map(lambda r: r[1]).collect()
            print('Found possible matches in our database: '
                  '{0}\n'.format([x for x in titles]))
            return movieIds

    def _append_ratings(self, userId, movieIds):
        """
        append a user's movie ratings to ratingsDF

        Parameter
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
        create a user with all movies except ones were rated for inferencing
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

    def _inference(self, model, fav_movie, n_recommendations):
        """
        return top n movie recommendations based on user's input movie

        Parameters
        ----------
        model: spark ALS model
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations

        Return
        ------
        list of top n similar movie recommendations
        """
        # create a userId
        userId = self.ratingsDF.agg({"userId": "max"}).collect()[0][0] + 1
        # get movieIds of favorite movies
        movieIds = self._regex_matching(fav_movie)
        # append new user with his/her ratings into data
        self._append_ratings(userId, movieIds)
        # matrix factorization
        model = model.fit(self.ratingsDF)
        # get data for inferencing
        inferenceDF = self._create_inference_data(userId, movieIds)
        # make inference
        return model.transform(inferenceDF) \
            .select(['movieId', 'prediction']) \
            .orderBy('prediction', ascending=False) \
            .rdd.map(lambda r: (r[0], r[1])) \
            .take(n_recommendations)

    def make_recommendations(self, fav_movie, n_recommendations):
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
            self._inference(self.model, fav_movie, n_recommendations)
        movieIds = [r[0] for r in raw_recommends]
        scores = [r[1] for r in raw_recommends]
        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))
        # get movie titles
        movie_titles = self.moviesDF \
            .filter(col('movieId').isin(movieIds)) \
            .select('title') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        # print recommendations
        print('Recommendations for {}:'.format(fav_movie))
        for i in range(len(movie_titles)):
            print('{0}: {1}, with rating '
                  'of {2}'.format(i+1, movie_titles[i], scores[i]))

class Dataset:
    """
    data object make loading raw files easier
    """
    def __init__(self, spark_session, filepath):
        """
        spark dataset constructor
        """
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.filepath = filepath
        # build spark data object
        self.RDD = self.load_file_as_RDD(self.filepath)
        self.DF = self.load_file_as_DF(self.filepath)

    def load_file_as_RDD(self, filepath):
        ratings_RDD = self.sc.textFile(filepath)
        header = ratings_RDD.take(1)[0]
        return ratings_RDD \
            .filter(lambda line: line != header) \
            .map(lambda line: line.split(",")) \
            .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))) # noqa


    def load_file_as_DF(self, filepath):
        ratings_RDD = self.load_file_as_rdd(filepath)
        ratingsRDD = ratings_RDD.map(lambda tokens: Row(userId=int(tokens[0]), movieId=int(tokens[1]), rating=float(tokens[2]))) # noqa
        return self.spark.createDataFrame(ratingsRDD)


def tune_ALS(model, train_data, validation_data, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data

    Parameters
    ----------
    model: spark ML model, ALS
    train_data: spark DF with columns ['userId', 'movieId', 'rating']
    validation_data: spark DF with columns ['userId', 'movieId', 'rating']
    maxIter: int, max number of learning iterations
    regParams: list of float, one dimension of hyper-param tuning grid
    ranks: list of float, one dimension of hyper-param tuning grid

    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # get ALS model
            als = model.setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            # train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse))
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
    movie_name = "GoldenEye (1995)"
    top_n = 10
    # initialize spark instance
    spark = SparkSession \
        .builder \
        .appName("movie recommender") \
        .getOrCreate()
    # initialiaze our recommender
    recommender = AlsRecommender(spark, movies_filename, ratings_filename)
    # initialize Hyperparameters for ALS
    recommender.set_model_params(10, 0.05, 20)
    # make recommendations
    recommender.make_recommendations(movie_name, top_n)
    # stop spark instance
    spark.stop()
