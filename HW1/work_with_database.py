import psycopg2
import pandas as pd
import pickle as pkl

CONN_PARAMS = {"dbname": 'ml_models',
               "user": 'ml_user',
               "password": 'password',
               "host": 'host_postgres',
               "port": 5432}


# psql -U ml_user -d ml_models

# conn = psycopg2.connect(**CONN_PARAMS)
# cursor = conn.cursor()


class DBModel(object):

    @staticmethod
    def create_table_if_not_exists() -> None:
        conn = psycopg2.connect(**CONN_PARAMS)
        with conn:
            with conn.cursor() as cur:
                # sql_cur.execute("""SELECT * FROM pg_catalog.pg_tables WHERE tablename = 'models';""")
                # sql_data = sql_cur.fetchall()
                # if len(sql_data) == 0:
                cur.execute("""CREATE TABLE IF NOT EXISTS MODELS
                                (
                                  MODELNAME varchar(100),
                                  MODELDATA BYTEA,
                                  MODELTYPE varchar(50),
                                  ISDELETED boolean
                                );""")

    @staticmethod
    def get_models_list() -> list:
        """
        Get name of models from BD
        :return:
            list of existing models

        """
        conn = psycopg2.connect(**CONN_PARAMS)
        with conn:
            with conn.cursor() as cur:
                cur.execute("""SELECT MODELNAME FROM MODELS WHERE MODELS.ISDELETED = FALSE;""")
                data = cur.fetchall()
        return pd.DataFrame(data, columns=["MODEL_NAME"])["MODEL_NAME"].tolist()

    # @staticmethod
    # def get_grouped_models() -> dict:
    #     """
    #     Get name of models from BD
    #     :return:
    #         models_list - list of existing models
    #         models_dct - {type_model: [model_names]}
    #     """
    #     conn = psycopg2.connect(**CONN_PARAMS)
    #     with conn:
    #         with conn.cursor() as cur:
    #             cur.execute("""SELECT MODEL_NAME, MODEL_TYPE FROM MODELS WHERE MODELS.IS_DELETED = FALSE""")
    #             data = cur.fetchall()
    #     return pd.DataFrame(data, columns=["MODEL_NAME", "MODEL_TYPE"]).groupby("MODEL_TYPE").groups

    @staticmethod
    def get_grouped_models_by_type(type_model: str) -> list:
        """
        Get name of models from BD
        :return:
            models_list - list of existing models
            models_dct - {type_model: [model_names]}
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
        conn = psycopg2.connect(**CONN_PARAMS)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT MODELDATA FROM MODELS WHERE MODELNAME = '{model_name}' AND MODELS.ISDELETED = FALSE""")
                data = cur.fetchone()
        return pkl.loads(data[0])

    @staticmethod
    def dump_model(model_name: str, model_data: dict) -> None:
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
        conn = psycopg2.connect(**CONN_PARAMS)
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"""UPDATE MODELS
                                SET ISDELETED = TRUE
                                WHERE MODELNAME = '{model_name}' AND ISDELETED = FALSE""")
