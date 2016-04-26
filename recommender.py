from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext
import math

def preparation(raw_user_movies, raw_hot_movies):
    user_id_stats = raw_user_movies.map(lambda line: line.split(',')[0]).distinct().zipWithUniqueId().map(lambda v: float(v[1])).stats()
    item_id_stats = raw_user_movies.map(lambda line: line.split(',')[1]).distinct().map(lambda v: float(v[1])).stats()
    user_score_stats = raw_user_movies.map(lambda line: float(line.split(',')[2])).stats()
    print user_id_stats
    print item_id_stats
    print user_score_stats
    movies_name = build_movies(raw_hot_movies)
    print '20645098', movies_name['20645098'].encode('utf-8')

def build_movies(raw_hot_movies):
    return raw_hot_movies.map(lambda line: line.split(',')).map(lambda tokens: (tokens[0], tokens[2])).collectAsMap()

def model(sc, raw_user_movies, raw_hot_movies):
    movies_name = build_movies(raw_hot_movies)
    user_id_to_int = raw_user_movies.map(lambda line: line.split(',')[0]).distinct().zipWithUniqueId().collectAsMap()
    print user_id_to_int['adamwzw']
    user_int_to_id = {v: k for k, v in user_id_to_int.iteritems()}
    rating_data = build_ratings(raw_user_movies, user_id_to_int)
    model = ALS.train(rating_data, 50, 10, 0.0001)
    print model.userFeatures().collect()[:2]
    for user_int in xrange(1, 30):
        check_recommend_result(user_int, raw_user_movies, movies_name, user_int_to_id, model)

def check_recommend_result(user_int, raw_user_movies, movies_name, user_int_to_id, model):
    user_id  = user_int_to_id[user_int]
    recommendations = model.recommendProducts(user_int, 5)
    user_see_movies = raw_user_movies.filter(lambda line: line.split(',')[0] == user_id)
    user_see_movies_name = user_see_movies.map(lambda line: line.split(',')).map(lambda tokens: (movies_name[tokens[1]], tokens[2])).collect()
    print user_id 
    print 'saw movies'
    for name, rating in user_see_movies_name:
        print name.encode('utf-8'), rating
    print 'recommend movies' 
    for recommendation in recommendations:
        if str(recommendation.product) in movies_name:
            print movies_name[str(recommendation.product)].encode('utf-8'), recommendation.rating

def evaluate(sc, raw_user_movies, raw_hot_movies):
    movies_name = build_movies(raw_hot_movies)
    user_id_to_int = raw_user_movies.map(lambda line: line.split(',')[0]).distinct().zipWithUniqueId().collectAsMap()
    ratings = build_ratings(raw_user_movies, user_id_to_int)
    num_iterations = 10
    for rank in [10, 50]:
        for lam in [1.0, 0.01, 0.0001]:
            model =  ALS.train(ratings, rank, num_iterations, lam)
            user_movies = ratings.map(lambda tokens: (tokens[0], tokens[1]))
            predictions = model.predictAll(user_movies).map(lambda r: ((r[0], r[1]), r[2]))
            print predictions.take(3)
            rates_and_preds = ratings.map(lambda tokens: ((tokens[0], tokens[1]), tokens[2])).join(predictions)
            print rates_and_preds.take(3)
            mse = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
            print "(rank:%d, lambda: %f,) Mean Squared Error = %f" % (rank, lam, mse)
    for rank in [10, 50]:
        for lam in [1.0, 0.01, 0.0001]:
            for alpha in [1.0, 40.0]:
                model = ALS.trainImplicit(ratings, rank, num_iterations, lam, alpha=alpha)
                user_movies = ratings.map(lambda tokens: (tokens[0], tokens[1]))
                predictions = model.predictAll(user_movies).map(lambda r: ((r[0], r[1]), r[2]))
                rates_and_preds = ratings.map(lambda tokens: ((tokens[0], tokens[1]), tokens[2])).join(predictions)
                print rates_and_preds.take(3)
                mse = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
                print "(rank:%d, lambda: %f, alpha: %f, implicit  ) Mean Squared Error = %f" % (rank, lam, alpha, mse)
def recommend(sc, raw_user_movies, raw_hot_movies):
    movies_name = build_movies(raw_hot_movies) 
    user_id_to_int = raw_user_movies.map(lambda line: line.split(',')[0]).distinct().zipWithUniqueId().collectAsMap()
    user_int_to_id = {v: k for k, v in user_id_to_int.iteritems()}
    ratings = build_ratings(raw_user_movies, user_id_to_int)
    #model.save(sc, "model")
    #model = MatrixFactorizationModel.load(sc, "model")
    model = ALS.train(ratings, 50, 10, 0.0001)
    def transform(record):
        user, recommendations = record[0], record[1]
        s = ''
        for r in recommendations:
            if str(r.product) in movies_name:
                s += str(r.product) + ':' + movies_name[str(r.product)].encode('utf-8') + ','
            else:
                s += str(r.product) + ':' + ','
        if s.endswith(","):
            s = s[:len(s) - 1]
        return (user_int_to_id[user], s)
    all_recommendations = model.recommendProductsForUsers(10).map(transform)
    for r in all_recommendations.take(10):
        print r[0], r[1]
    all_recommendations.saveAsTextFile("result.csv")
def build_ratings(raw_user_movies, user_id_to_int):
    def transform(line):
        values = line.split(',')
        values[0] = user_id_to_int[values[0]]
        if values[2] == '-1':
            values[2] = 3
        return values[0], int(values[1]), int(values[2])
    return raw_user_movies.map(transform)

sc = SparkContext("local", "Simple Recommend")
raw_user_movies = sc.textFile("user_movies.csv")
raw_hot_movies = sc.textFile("hot_movies.csv")

print raw_user_movies.first()
#print raw_hot_movies.first().encode('utf-8')
preparation(raw_user_movies, raw_hot_movies)
#evaluate(sc, raw_user_movies, raw_hot_movies)
#model(sc, raw_user_movies, raw_hot_movies)
recommend(sc, raw_user_movies, raw_hot_movies)


