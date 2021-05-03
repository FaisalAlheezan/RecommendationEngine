import numpy as np
import pandas as pd
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.sql('''select 'spark' as hello ''')
df.show()
import findspark
findspark.init()
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession ,Row
from pyspark.sql.functions import col, explode
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Creating a spark session is needed to run spark dataframes
spark = SparkSession.builder.appName('abc').getOrCreate()

# The Recommednation engine class
class RecommendationEngine:
    # Training the model
    def __train_model(self):
        logger.info("Training the Model")
        # For the explicit version
        # cv = CrossValidator(estimator=als3, estimatorParamMaps=paramGrid, evaluator=evaluator3, numFolds=5)
        # self.model3 = cv.fit(self.spark_df)

        # For the implicit version
        als3= ALS(nonnegative=True, rank=20, regParam=0.01, \
                   userCol="user_id_maped", implicitPrefs=True, itemCol="content_id", ratingCol="rating",
                   coldStartStrategy="drop")
        self.model3 = als3.fit(self.spark_df)
        logger.info("ALS model built!")

    # The initialization
    def __init__(self,dataset_spark, dataset_content):
        logger.info("Starting the engine")
        # Reading dataset2 and df_content for pandas dataframe
        self.panda_df = pd.read_csv(dataset_spark)
        self.panda_df = self.panda_df[['user_id_maped', 'content_id', 'rating']]
        self.panda_cont = pd.read_csv(dataset_content)

        # Converting the pandas dataframes to spark dataframe so they can be executed
        self.spark_df = spark.createDataFrame(self.panda_df)
        self.spark_cont = spark.createDataFrame(self.panda_cont[['original_name','content_id', 'program_genre', 'program_class']])
        #Running the model for first time
        # self.__train_model()

    # adding user data
    def add_ratings(self,content_name,rating):
        # adding user data as user 500000. so that the user wouldn't have to do it himself
        user_id_l = [500000]
        user_id_list = len(rating)* user_id_l

        content_id = []
        # clearing the space between the name of the movie or show
        stripped_content_name = [s.strip() for s in content_name]
        # Turning the name of the movie or the show into an content_id
        for i in stripped_content_name:
            temp = self.panda_cont[self.panda_cont['original_name'] == i]['content_id'].values[0]
            content_id.append(temp)
        logger.info("ALS model built!")

        # creating a 2d array from the user input
        arrayn = np.array([user_id_list, content_id, rating])
        # turning said array into a pandas dataframe to add it to dataset2
        ac = pd.DataFrame(arrayn.T,columns=['user_id_maped', 'content_id', 'rating'])
        ac['user_id_maped'] = ac['user_id_maped'].astype(int)
        ac['content_id'] = ac['content_id'].astype(int)
        ac['rating'] = ac['rating'].astype(int)

        # adding the generated dataframe to dataset2 and training the model
        self.spark_df = spark.createDataFrame(pd.concat([self.panda_df, ac],ignore_index=True))
        self.__train_model()


        # Generating the user recommendation and turning into a pandas dataframe then sending to the user
        logger.info("Generating rating for user...")
        recommendations = self.model3.recommendForAllUsers(10)
        recommendations = recommendations \
            .withColumn("rec_exp", explode("recommendations")) \
            .select('user_id_maped', col("rec_exp.content_id"), col("rec_exp.rating"))
        rating = recommendations.join(self.spark_cont, on='content_id').filter('user_id_maped = 500000 ') \
            .sort('rating', ascending=False).toPandas()

        return rating
