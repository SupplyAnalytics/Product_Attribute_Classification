
def data_dump():
    import mysql.connector
    import pandas as pd
    mydb=mysql.connector.connect(host='report-reader.cluster-custom-c8oe0gvszktr.ap-south-1.rds.amazonaws.com',
                            user='skreadonlyuser_api',
                            password='XUerT9JxnWnEstSA')
    cursor=mydb.cursor()
    cursor.execute('''
    Select Distinct
    v.variantId as variantId,
    upper(clr.colorName) as color,
    clr.colorHex as colorHex,
    case when Upper(cm.name) = 'T-SHIRT' then 'T-SHIRTS' else Upper(cm.name) end as category,
    Upper(cm1.name) as mainCategory,
    Upper(cm2.name) as superCategory,
    pa.attribute as productAttribute
    from shoekonnect_live.variants as v 
    INNER JOIN shoekonnect_live.colors as clr on clr.uniqueId = v.colorId
    INNER JOIN shoekonnect_live.products as p on p.productid = v.productid 
    INNER JOIN shoekonnect_live.category_master as cm on cm.categoryid = p.subcategoryid
    INNER JOIN shoekonnect_live.category_master as cm1 on cm1.categoryid = cm.parentid
    INNER JOIN shoekonnect_live.category_master as cm2 on cm2.categoryid = cm1.parentid
    INNER JOIN shoekonnect_live.products_attribute as pa on pa.productid = p.productid
    where pa.attribute IS NOT NULL
    ''')
    print('Data Fetched from DB')
    import pandas as pd
    import json
    rows=cursor.fetchall()
    columns=[i[0] for i in  cursor.description]
    df_base=pd.DataFrame(rows,columns=columns)
    df_base.to_csv('Variants_Base_Data.csv')
    print('Data Downloaded in CSV')

data_dump()