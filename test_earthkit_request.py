import earthkit.data as ekd

# request_d = dict(
#     variable=['sea_surface_salinity'],
#     # area=[40, -100, 0, -120],
#     year=['1996'],
#     month=['01', '02'],
#     product_type="consolidated",
#     vertical_resolution="single_level",
#     # grid=[.1, .1],
#     # format="netcdf"
# ),

request_d = dict(
    param=['2t'],
    # area=[40, -100, 0, -120],
    # date="2023-05-10"
    # grid=[.1, .1],
    # format="netcdf"
),

ds_chunk = ekd.from_source("ecmwf-open-data", request_d).to_xarray()
# ds_chunk = ekd.from_source("cds", "reanalysis-oras5", request_d).to_xarray()
ds_chunk.to_netcdf('prova.nc')

# import os
# import certifi

# # Ensure SSL and Requests use certifi CA bundle
# os.environ["SSL_CERT_FILE"] = certifi.where()
# os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# import requests
# print("Using bundle:", os.environ["REQUESTS_CA_BUNDLE"])
# url = "https://object-store.os-api.cci2.ecmwf.int"
# r = requests.get(url, timeout=10)
# print("Status:", r.status_code)