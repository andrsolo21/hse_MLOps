

psql -U ml_user -d ml_models

select * from models;

drop table MODELS;

SELECT MODELNAME FROM MODELS WHERE MODELS.IS_DELETED = FALSE;

SELECT "MODELNAME" FROM MODELS WHERE "ISDELETED" = FALSE AND "MODELTYPE" = 'Lasso';

SELECT * FROM pg_catalog.pg_tables WHERE tablename = 'models';
