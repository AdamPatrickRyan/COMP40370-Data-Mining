import pandas as pd
import numpy as np
import sqlalchemy as sqla
import datetime as dt
import psycopg2 as psy
from psycopg2 import sql

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%matplotlib notebook

def question_two():
    """Question 2"""
    
        
    def draw_cube(df,fig_name):
        
        if 'Status' not in df.columns:
            raise ValueError("Status Not Present")

        df.loc[df['Status'] == 'Technician', 'Status_cat'] = 1
        df.loc[df['Status'] == 'Senior Technician', 'Status_cat'] = 2
        df.loc[df['Status'] == 'Deputy Director', 'Status_cat'] = 3
        df.loc[df['Status'] == 'Director', 'Status_cat'] = 4
        
        x = df['Year of Birth']
        y = df['Salary']
        z = df['Status_cat']

        fig=plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Salary - Status - Y.O.B Data Cube')
        ax.set_xlabel('Status Category')
        ax.set_ylabel('Year of Birth')
        ax.set_zlabel('Salary')
        plt.xticks(range(1,5),labels=['Technical','Senior Technicial','Deputy Director','Director'])
        ax.scatter3D(z, x, y, c=z)
        
        plt.savefig(fig_name)
        
        
        return
    
    
    #Read it
    df=pd.read_csv('./specs/DW_dataset.csv')
    
    #Fix errors
    df.columns=['Emp ID', 'Name', 'Year of Birth', 'Gender', 'Status', 'Salary']
    
    #Remove euro and set nulls
    df['Salary']=df['Salary'].str.replace("€ ","")
    df['Salary']=df['Salary'].replace(" ", np.nan)
    df['Status']=df['Status'].str.strip()
    
    df['Salary']=pd.to_numeric(df['Salary'])
    df['Salary']=df['Salary'].fillna(3300)
    
    #Add Age Attribute
    df['Age']=dt.datetime.now().year - df['Year of Birth']

    #Plot
    df.plot(kind='scatter',x='Age',y='Salary',title='Age vs Salary')
    plt.gcf()
    plt.savefig('./outputs/age_outlier.png')

    draw_cube(df,'./outputs/3d_cube_Q2P4.png')
    
    return df


def question_three(username='adamryan',password='',server='localhost',db='local_db'):
    
    
    def write_record(name,details,engine,username=username,password=password,db=db):
        """
        Write a record to a table with:
        name=table name
        details=dictionary with {column name:value}
        engine=database engine.
        
        """

        
   
        #Protect against SQL Injection - Works for any Table
        Insert_SQL = sql.SQL("insert into {}"" ({}) values ({})").format(
            sql.Identifier(name)
            ,sql.SQL(', ').join(map(sql.Identifier, details.keys()))
            ,sql.SQL(', ').join(sql.Placeholder() * len(details.keys()))
            )
            
        
        values = tuple(details.values())
        
        with engine.connect() as connection:
                
            connection.execute(Insert_SQL.as_string(psy.connect("dbname={} user={} password={}".format(db,username,password))),values)

            
        return
        
    
    def read_record(field,table_name,detail,engine,username=username,password=password,db=db):
        """Read a field from a table given a value
        
        Input:
        field: Field to read
        table_name: Name of table
        detail: Dictionary of Length 1 {column:Value}
        engine:database engine
        """
        
        if type(detail)!=dict or len(detail)>1:
            raise ValueError("detail needs to be a dictionary with the column name and value")
            
        for key,value in detail.items():
            filter_field=key
            value_field=value
        
        
        Select_SQL=sql.SQL("""SELECT 
                            {field}
                        FROM 
                            {table_name} 
                        WHERE 
                            {filter_field} = %s""").format(
                                                field=sql.Identifier(field)
                                                ,table_name=sql.Identifier(table_name)
                                                ,filter_field=sql.Identifier(filter_field)
                                                        )
        
        return_value=None
        
        with engine.connect() as connection:
            result=connection.execute(Select_SQL.as_string(psy.connect("dbname={} user={} password={}".format(db,username,password))),(value_field,))
            return_value=result.first()[0]
        
        return return_value

    def update_record(field,table_name,detail,new_value,engine,username=username,password=password,db=db):
        """Update a field from a table given a value
        
        Input:
        field: Field to read
        table_name: Name of table
        detail: Dictionary of Length 1 {column:Value}
        new_value: New Value
        engine:database engine
        """
        
        
        if type(detail)!=dict or len(detail)>1:
            raise ValueError("detail needs to be a dictionary with the column name and value")
            
        for key,value in detail.items():
            filter_field=key
            value_filter_field=value
        
        
        
        
        
        
        Update_SQL=sql.SQL("""UPDATE
                                {table_name}
                        SET 
                            {column_to_update}=%s
                        WHERE 
                            {filter_field} = %s""").format(
                                                column_to_update=sql.Identifier(field)
                                                ,table_name=sql.Identifier(table_name)
                                                ,filter_field=sql.Identifier(filter_field)
                                                        )
                          
            
        with engine.connect() as connection:
            connection.execute(Update_SQL.as_string(psy.connect("dbname={} user={} password={}".format(db,username,password))),(new_value,value_filter_field,))
            
        return

    def write_dataset(name,dataset,engine):
        """Insert a dataframe as a table. Replaces if exists.
        
        Input:
        Name of table
        Dataframe
        Engine"""
        
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Not a dataframe!")
        
        dataset.to_sql(name, engine, index=False, if_exists='replace')
        
        return

    def read_dataset(name,engine):
        """Read dataframe given table name"""
        
        try:
            dataset = pd.read_sql_table(name,engine)
            
        except:
            dataset = pd.DataFrame([])
            
        return dataset

    def list_datasets(engine):
        
        
        datasets = engine.execute("""SELECT 
                                        table_name 
                                    FROM 
                                        INFORMATION_SCHEMA.TABLES 
                                    WHERE 
                                        table_schema = 'public' 
                                    ORDER BY 
                                        table_name;""")
        
        table_list=[]
        for table in datasets.fetchall():
            table_list+=table
        
        dataset_dictionary=dict({})
        
        for table in table_list:
            dataset_dictionary[table]=read_dataset(table,engine)
            
                
        return dataset_dictionary
    
    engine = sqla.create_engine('postgresql+psycopg2://{}:{}@{}/{}'.format('adamryan','','localhost','local_db'))
    
    sample_dataset=pd.read_csv('./specs/input_DW_data.csv')
    
    
    table_name='input_dw_data'
    table_columns=sample_dataset.columns
    
    #Write
    print("Write Dataset:\n")
    write_dataset(name=table_name
                  ,dataset=sample_dataset
                  ,engine=engine)
    
    #Read 
    print("Read Dataset:\n")
    db_dataset=read_dataset(name=table_name
                       ,engine=engine)
    print(db_dataset)
    
    max_id=5
    max_id=max(db_dataset['id'])
    
    #Insert
    print("Insert:\n")
    write_record(name=table_name
                ,details={
                    'id':max_id+1
                    ,'first_name':'New'
                    ,'middle_name':'Sample'
                    ,'last_name':'Data'
                    ,'favourite_number':max_id+1
                    ,'location':'Ireland'
                }
                 ,engine=engine)
    
    #Read Dataset again after insert:
    db_dataset=read_dataset(name=table_name
                       ,engine=engine)
    print(db_dataset)
    
    max_id=max(db_dataset['id'])
    
    #Read each column in max id
    print("Reading Record:\n")
    for column in table_columns:
        print("""
{}:{}
""".format(column,read_record(field=column
                                            ,table_name=table_name
                                            ,detail=dict({'id':max_id})
                                            ,engine=engine)))
        
    print("Updating Record:\n")
    update_record(field='location'
                  ,table_name=table_name
                  ,detail={'id':max_id}
                  ,new_value='UpdateLocation'
                  ,engine=engine)
    
    
    #Read 
    print("Read Dataset After update:\n")
    db_dataset=read_dataset(name=table_name
                       ,engine=engine)
    
    max_id=max(db_dataset['id'])
    
    print(db_dataset)
    
    #List all Tables
    
    dataset_dict=list_datasets(engine=engine)
    
    return dataset_dict


def run():
    question_two()
    try:
        question_three()
    except:
        print("You need to have the same setup as me if you want to properly run this without adjusting your database paramters")

run()