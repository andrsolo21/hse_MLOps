CREATE TABLE models
(
  id serial NOT NULL,
  MODEL_NAME varchar(100),
  MODEL_DATA text,
  MODEL_TYPE varchar(50),
  is_deleted boolean
);

psql -U ml_user -d ml_models

select * from models;

drop table MODELS;

SELECT MODEL_NAME FROM MODELS WHERE MODELS.IS_DELETED = FALSE;

SELECT "MODEL_NAME" FROM MODELS WHERE "IS_DELETED" = FALSE AND "MODEL_TYPE" = 'Lasso';

SELECT * FROM pg_catalog.pg_tables WHERE tablename = 'models';
