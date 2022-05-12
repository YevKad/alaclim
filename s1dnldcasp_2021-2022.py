import time
import os
from datetime import datetime, timedelta
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt

s1dir='D:/images/s1/'
usr='iceman.kz'#'yevkad'
pwd='Polynya25'#'ASkdas1456x'
api = SentinelAPI(usr,pwd,'https://apihub.copernicus.eu/apihub/')
print ('OK')
# download single scene by known product id
footprintfl='footprint.geojson'

dt_start='20220319'
dt_end='20220416'
# search by polygon, time, and SciHub query keywords
while True:
    try:
        for i in range(5):

            footprint = geojson_to_wkt(read_geojson(footprintfl))
            products = api.query(footprint,
                                 (dt_start, dt_end),
                                 platformname = 'Sentinel-1',
                                 producttype='GRD' )

            # download all results from the search
            #print(products)
            keys=list(products)

            for k in keys:
                print(products[k]['filename'])


                print('downloading...')
                api.download(k,directory_path=s1dir)
        print('sleeping')
        time.sleep(600)
    except Exception as e:
        print(e)
