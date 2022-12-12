import psycopg2
import pandas as pd
import pickle as pkl

CONN_PARAMS = {"dbname": 'ml_models',
               "user": 'ml_user',
               "password": 'password',
               "host": 'host_postgres',
               "port": 5432}


class DBModel(object):

    @staticmethod
    def create_table_if_not_exists() -> None:
        """
        Create table in DataBase if it not exists
        :return: None
        """
        conn = psycopg2.connect(**CONN_PARAMS)
        with conn:
            with conn.cursor() as cur:
                cur.execute("""CREATE TABLE IF NOT EXISTS MODELS
                                (
                                  MODELNAME varchar(100),
                                  MODELDATA BYTEA,
                                  MODELTYPE varchar(50),
                                  ISDELETED boolean
                                );""")

    @staticmethod
    def get_models_list() -> list[str]:
        """
        Get name of models from DataBase
        :return: list of models names
        """
        conn = psycopg2.connect(**CONN_PARAMS)
        with conn:
            with conn.cursor() as cur:
                cur.execute("""SELECT MODELNAME FROM MODELS WHERE MODELS.ISDELETED = FALSE;""")
                data = cur.fetchall()
        return pd.DataFrame(data, columns=["MODEL_NAME"])["MODEL_NAME"].tolist()

    @staticmethod
    def get_grouped_models_by_type(type_model: str) -> list[str]:
        """
        Get name of models from DataBase with given type
        :param type_model: type of models
        :return: list of models names
        """
        conn = psycopg2.connect(**CONN_PARAMS)
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"""SELECT MODELNAME FROM MODELS WHERE MODELS.ISDELETED = 
                            FALSE AND MODELS.MODELTYPE = '{type_model}'""")
                data = cur.fetchall()
        return pd.DataFrame(data, columns=["MODEL_NAME"])["MODEL_NAME"].tolist()

    @staticmethod
    def get_model(model_name: str) -> dict:
        """
        Get data for creating ML model
        :param model_name: model name
        :return: data fot creating model
        """
        conn = psycopg2.connect(**CONN_PARAMS)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT MODELDATA FROM MODELS WHERE MODELNAME = '{model_name}' AND MODELS.ISDELETED = FALSE""")
                data = cur.fetchone()
        return pkl.loads(data[0])

    @staticmethod
    def dump_model(model_name: str, model_data: dict) -> None:
        """
        save model to DataBase
        :param model_name: model name
        :param model_data: data for store
        :return: None
        """
        models_list = DBModel.get_models_list()
        conn = psycopg2.connect(**CONN_PARAMS)
        with conn:
            with conn.cursor() as cur:
                if model_name in models_list:
                    cur.execute("""UPDATE MODELS
                                    SET MODELDATA = %s
                                    WHERE MODELNAME = %s AND MODELS.ISDELETED = FALSE""", (
                        pkl.dumps(model_data),
                        model_name
                    ))
                else:
                    cur.execute("""INSERT INTO MODELS
                                    VALUES (%s,
                                            %s,
                                            %s,
                                            FALSE)""", (model_name,
                                                        pkl.dumps(model_data),
                                                        model_name.split("_")[0]))

    @staticmethod
    def delete_model(model_name: str) -> None:
        """
        Mark model as deleted
        :param model_name: model name
        :return: None
        """
        conn = psycopg2.connect(**CONN_PARAMS)
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"""UPDATE MODELS
                                SET ISDELETED = TRUE
                                WHERE MODELNAME = '{model_name}' AND ISDELETED = FALSE""")
